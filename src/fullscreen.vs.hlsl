struct vs_output
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

vs_output main(uint vertex_id : SV_VertexID)
{
    float2 positions[6] = {float2(-1.0, -1.0), float2(1.0, 1.0),  float2(1.0, -1.0),
                           float2(-1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0)};

    vs_output output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = positions[vertex_id] * float2(0.5, -0.5) + 0.5;
    return output;
}
