Shader "Custom/Grass"
{
    Properties
    {
        // Color at the base of each blade.
        _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        
        // Color at the tip of each blade.
        _TipColor("Tip Color", Color) = (1, 1, 1, 1)
        
        // Optional texture for each blade (e.g., for detail or patterns).
        _BladeTexture("Blade Texture", 2D) = "white" {}

        // Minimum and maximum blade widths in local space.
        _BladeWidthMin("Blade Width (Min)", Range(0, 0.1)) = 0.02
        _BladeWidthMax("Blade Width (Max)", Range(0, 0.1)) = 0.05

        // Minimum and maximum blade heights in local space.
        _BladeHeightMin("Blade Height (Min)", Range(0, 2)) = 0.1
        _BladeHeightMax("Blade Height (Max)", Range(0, 2)) = 0.2

        // Number of vertical segments that make up each blade.
        _BladeSegments("Blade Segments", Range(1, 10)) = 3

        // How far the top of the blade can bend forward.
        _BladeBendDistance("Blade Forward Amount", Float) = 0.38

        // Exponent controlling how strongly the blade curves (e.g., near the tip).
        _BladeBendCurve("Blade Curvature Amount", Range(1, 4)) = 2

        // Adds random variation to the bending between individual blades.
        _BendDelta("Bend Variation", Range(0, 1)) = 0.2

        // Controls how large an edge can be before we tessellate it more (smaller = more tessellation).
        _TessellationGrassDistance("Tessellation Grass Distance", Range(0.01, 2)) = 0.1

        // A texture that determines where grass should appear (white = grass, black = no grass).
        _GrassMap("Grass Visibility Map", 2D) = "white" {}
        // Threshold in the GrassMap (if pixel < threshold => no grass).
        _GrassThreshold("Grass Visibility Threshold", Range(-0.1, 1)) = 0.5
        // Smoothly blends from no grass to grass above threshold.
        _GrassFalloff("Grass Visibility Fade-In Falloff", Range(0, 0.5)) = 0.05

        // A texture that adds wind offset/noise.
        _WindMap("Wind Offset Map", 2D) = "bump" {}
        // A vector describing the wind direction and strength.
        _WindVelocity("Wind Velocity", Vector) = (1, 0, 0, 0)
        // Controls how fast the wind cycles/pulses over time.
        _WindFrequency("Wind Pulse Frequency", Range(0, 1)) = 0.01
    }

    SubShader
    {
        // Tags control the rendering order and pipeline usage (URP in this case).
        Tags
        {
            "RenderType" = "Opaque"
            "Queue" = "Geometry"
            "RenderPipeline" = "UniversalPipeline"
        }
        LOD 100
        Cull Off // We disable culling so both sides of the grass are visible.

        HLSLINCLUDE

            // Include necessary URP core and lighting libraries.
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // Some mathematical constants for convenience.
            #define UNITY_PI 3.14159265359f
            #define UNITY_TWO_PI 6.28318530718f

            // Default number of segments.
            #define BLADE_SEGMENTS 4

            // Declare a "Constant Buffer" of material parameters (matches the Properties above).
            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _TipColor;
                sampler2D _BladeTexture;

                float _BladeWidthMin;
                float _BladeWidthMax;
                float _BladeHeightMin;
                float _BladeHeightMax;

                float _BladeBendDistance;
                float _BladeBendCurve;

                float _BendDelta;

                float _TessellationGrassDistance;
                
                sampler2D _GrassMap;
                float4 _GrassMap_ST;
                float  _GrassThreshold;
                float  _GrassFalloff;

                sampler2D _WindMap;
                float4 _WindMap_ST;
                float4 _WindVelocity;
                float  _WindFrequency;

            CBUFFER_END

            // Input to the initial vertex stage from the mesh.
            struct VertexInput
            {
                float4 vertex  : POSITION;  // The vertex position (object space).
                float3 normal  : NORMAL;    // The vertex normal (object space).
                float4 tangent : TANGENT;   // Tangent + sign for bitangent.
                float2 uv      : TEXCOORD0; // Primary UV channel.
            };

            // Output from our vertex stages to the next pipeline stages.
            struct VertexOutput
            {
                float4 vertex  : SV_POSITION; // Position for the next stage (often world or clip).
                float3 normal  : NORMAL;      // Normal (we may transform to world space).
                float4 tangent : TANGENT;     // Tangent plus sign for bitangent.
                float2 uv      : TEXCOORD0;   // UV used for sampling textures.
            };

            // After geometry expansion, we pass data to the fragment shader in this struct.
            struct GeomData
            {
                float4 pos : SV_POSITION;    // Final clip-space position of the new geometry.
                float2 uv  : TEXCOORD0;      // UV for texturing each blade.
                float3 worldPos : TEXCOORD1; // The blade vertex in world space (for shading, etc.).
            };

            // Holds the edge and inside tessellation factors for patches (triangles).
            struct TessellationFactors
            {
                float edge[3] : SV_TessFactor;     // Edge tessellation factors.
                float inside  : SV_InsideTessFactor; // Inside tess factor for triangular patches.
            };

            //--------------------------------------------
            // Utility Functions
            //--------------------------------------------

            // A pseudo-random function returning 0..1, used for random blade orientation.
            float rand(float3 co)
            {
                return frac(sin(dot(co.xyz, float3(12.9898, 78.233, 53.539))) * 43758.5453);
            }

            // Builds a 3x3 rotation matrix around "axis" by "angle".
            float3x3 angleAxis3x3(float angle, float3 axis)
            {
                float c, s;
                sincos(angle, s, c);

                float t = 1 - c;
                float x = axis.x;
                float y = axis.y;
                float z = axis.z;

                return float3x3
                (
                    t * x * x + c,     t * x * y - s * z, t * x * z + s * y,
                    t * x * y + s * z, t * y * y + c,     t * y * z - s * x,
                    t * x * z - s * y, t * y * z + s * x, t * z * z + c
                );
            }

        ENDHLSL

        // We specify the pipeline stages: vertex -> hull -> domain -> geometry -> fragment.
        Pass
        {
            Name "GrassPass"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM

            // We require geometry-shader capability in this pass.
            #pragma require geometry

            // The order we define the stages: vertex -> hull -> domain -> geometry -> fragment.
            #pragma vertex   geomVertex
            #pragma hull     hull
            #pragma domain   domain
            #pragma geometry geom
            #pragma fragment frag

            // 1) Vertex Shader: transforms mesh vertices from object to world space.
            VertexOutput geomVertex (VertexInput v)
            {
                VertexOutput o; 
                
                // Convert object-space position to world space.
                o.vertex = float4(TransformObjectToWorld(v.vertex), 1.0f);

                // Convert object-space normal to world space normal.
                o.normal = TransformObjectToWorldNormal(v.normal);

                // Pass the tangent unchanged (still in object space).
                o.tangent = v.tangent;

                // Apply the _GrassMap scale/offset to the UV.
                o.uv = TRANSFORM_TEX(v.uv, _GrassMap);

                return o;
            }

            // 2) An extra vertex shader for tessellation input (just passes data along).
            VertexOutput tessVert(VertexInput v)
            {
                VertexOutput o;
                o.vertex  = v.vertex;
                o.normal  = v.normal;
                o.tangent = v.tangent;
                o.uv      = v.uv;
                return o;
            }

            // Computes how much we tessellate each edge based on distance.
            float tessellationEdgeFactor(VertexInput vert0, VertexInput vert1)
            {
                float3 v0 = vert0.vertex.xyz;
                float3 v1 = vert1.vertex.xyz;
                // Larger distance => bigger factor => more tessellation.
                float edgeLength = distance(v0, v1);
                return edgeLength / _TessellationGrassDistance;
            }

            // Calculates tessellation factors for each patch (triangle).
            TessellationFactors patchConstantFunc(InputPatch<VertexInput, 3> patch)
            {
                TessellationFactors f;

                // For each edge, compute how many subdivisions we want.
                f.edge[0] = tessellationEdgeFactor(patch[1], patch[2]);
                f.edge[1] = tessellationEdgeFactor(patch[2], patch[0]);
                f.edge[2] = tessellationEdgeFactor(patch[0], patch[1]);

                // The inside factor is the average of the three edges, for a uniform subdivision.
                f.inside = (f.edge[0] + f.edge[1] + f.edge[2]) / 3.0f;

                return f;
            }

            // Hull Shader: outputs the original 3 control points of the triangle.
            // The [patchconstantfunc] above is used for the tessellation factors.
            [domain("tri")]
            [outputcontrolpoints(3)]
            [outputtopology("triangle_cw")]
            [partitioning("integer")]
            [patchconstantfunc("patchConstantFunc")]
            VertexInput hull(InputPatch<VertexInput, 3> patch, uint id : SV_OutputControlPointID)
            {
                // Simply return the vertex from the patch to pass on to the domain shader.
                return patch[id];
            }

            // Domain Shader: invoked for each tessellated vertex. Interpolates the corners
            // with barycentric coordinates to get a new position, normal, etc.
            [domain("tri")]
            VertexOutput domain(TessellationFactors factors, 
                                OutputPatch<VertexInput, 3> patch, 
                                float3 barycentricCoordinates : SV_DomainLocation)
            {
                VertexInput i;

                // Macro to interpolate a field from the three patch vertices
                #define INTERPOLATE(fieldname) \
                    i.fieldname = patch[0].fieldname * barycentricCoordinates.x \
                                + patch[1].fieldname * barycentricCoordinates.y \
                                + patch[2].fieldname * barycentricCoordinates.z;

                INTERPOLATE(vertex)
                INTERPOLATE(normal)
                INTERPOLATE(tangent)
                INTERPOLATE(uv)

                // Pass those interpolated values through tessVert (which builds a VertexOutput).
                return tessVert(i);
            }

            // Helper function to transform a point from local offset space to clip space, 
            // storing UV and worldPos for the fragment stage.
            GeomData TransformGeomToClip(float3 pos, float3 offset, float3x3 transformationMatrix, float2 uv)
            {
                GeomData o;

                // Move from 'pos' by 'offset' in the local blade space, then transform to clip space.
                o.pos = TransformObjectToHClip(pos + mul(transformationMatrix, offset));
                // Pass along the UV for coloring in fragment shader.
                o.uv = uv;
                // Also store the world-space position (for advanced lighting, shadows, etc.).
                o.worldPos = TransformObjectToWorld(pos + mul(transformationMatrix, offset));

                return o;
            }

            // Geometry Shader: expands each tessellated vertex into a custom grass blade (triangle strip).
            [maxvertexcount(BLADE_SEGMENTS * 2 + 1)]
            void geom(point VertexOutput input[1], inout TriangleStream<GeomData> triStream)
            {
                // Sample GrassMap at the current UV to decide if grass is allowed (visibility).
                float grassVisibility = tex2Dlod(_GrassMap, float4(input[0].uv, 0, 0)).r;

                // If the grass map is below threshold, skip generating a blade here.
                if (grassVisibility <= _GrassThreshold) return;

                // The position (in world space) from the vertex stage.
                float3 pos = input[0].vertex.xyz;
                float3 normal = input[0].normal;
                float4 tangent = input[0].tangent;

                // Recompute bitangent (perpendicular to normal & tangent).
                float3 bitangent = cross(normal, tangent.xyz) * tangent.w;

                // Build a matrix that transforms from the mesh tangent space into "blade local space".
                float3x3 tangentToLocal = float3x3
                (
                    tangent.x,    bitangent.x,    normal.x,
                    tangent.y,    bitangent.y,    normal.y,
                    tangent.z,    bitangent.z,    normal.z
                );

                // Create a random rotation around the "z-axis" in this tangent space for variety.
                float3x3 randRotMatrix = angleAxis3x3(rand(pos) * UNITY_TWO_PI, float3(0, 0, 1.0f));

                // Add random bending around the bottom of the blade (for more variation).
                float3x3 randBendMatrix = angleAxis3x3(rand(pos.zzx) * _BendDelta * UNITY_PI * 0.5f, float3(-1.0f, 0, 0));

                // Sample a wind map using pos.xz, plus time and wind velocity, 
                // to get a dynamic offset.
                float2 windUV = pos.xz * _WindMap_ST.xy + _WindMap_ST.zw 
                                + normalize(_WindVelocity.xzy) * _WindFrequency * _Time.y;
                
                // Convert the wind texture's -1..1 range to a tilt magnitude.
                float2 windSample = (tex2Dlod(_WindMap, float4(windUV, 0, 0)).xy * 2 - 1) 
                                     * length(_WindVelocity);

                // Build a rotation axis from windSample, then compute rotation.
                float3 windAxis = normalize(float3(windSample.x, windSample.y, 0));
                float3x3 windMatrix = angleAxis3x3(UNITY_PI * windSample, windAxis);

                // Combine tangent-space transform with random rotation & wind rotation
                // for the bottom (base) and top (tip) of the blade.
                float3x3 baseTransformationMatrix = mul(tangentToLocal, randRotMatrix);
                float3x3 tipTransformationMatrix  = mul(mul(mul(tangentToLocal, windMatrix), 
                                                     randBendMatrix), randRotMatrix);

                // Gradually fade from no grass to grass based on the GrassMap above threshold.
                float falloff = smoothstep(_GrassThreshold, _GrassThreshold + _GrassFalloff, grassVisibility);

                // Randomly pick a width, height, and forward bend for each blade, 
                // scaled by the 'falloff' factor for smooth transitions.
                float width   = lerp(_BladeWidthMin,  _BladeWidthMax,  rand(pos.xzy) * falloff);
                float height  = lerp(_BladeHeightMin, _BladeHeightMax, rand(pos.zyx) * falloff);
                float forward = rand(pos.yyz) * _BladeBendDistance;

                // The blade is made up of _BladeSegments slices + one tip. 
                // Each loop iteration appends two vertices forming a horizontal strip.
                for (int i = 0; i < BLADE_SEGMENTS; ++i)
                {
                    float t = i / (float)BLADE_SEGMENTS;

                    // The offset moves from full width at the bottom to near 0 at the top.
                    // Also bend the blade forward and raise it up (height * t).
                    float3 offset = float3(width * (1 - t), pow(t, _BladeBendCurve) * forward, height * t);

                    // Use base transform near the bottom, tip transform near the top.
                    float3x3 transformationMatrix = (i == 0) ? baseTransformationMatrix 
                                                             : tipTransformationMatrix;

                    // Append the left side of the strip.
                    triStream.Append(TransformGeomToClip(pos,
                                           float3( offset.x, offset.y, offset.z),
                                           transformationMatrix,
                                           float2(0, t)));
                    // Append the right side of the strip.
                    triStream.Append(TransformGeomToClip(pos,
                                           float3(-offset.x, offset.y, offset.z),
                                           transformationMatrix,
                                           float2(1, t)));
                }

                // Add a single tip vertex in the center at the top (0.5 in UV.x).
                triStream.Append(TransformGeomToClip(pos,
                                     float3(0, forward, height),
                                     tipTransformationMatrix,
                                     float2(0.5, 1)));

                // Restart the triangle strip so each blade is separate.
                triStream.RestartStrip();
            }

            // Fragment Shader: takes the expanded geometry data and computes final color.
            float4 frag(GeomData i) : SV_Target
            {
                // Sample the blade texture at the UV (if it's white, it won't change color).
                float4 color = tex2D(_BladeTexture, i.uv);

                // Blend between the base color (bottom) and tip color (top)
                // based on i.uv.y which goes from 0..1 along the blade's length.
                return color * lerp(_BaseColor, _TipColor, i.uv.y);
            }

            ENDHLSL
        }
    }
}
