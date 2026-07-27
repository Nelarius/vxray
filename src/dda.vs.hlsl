struct vs_output
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

vs_output main(uint const vertex_id : SV_VertexID)
{
    float2 const positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
    };

    vs_output output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = positions[vertex_id] * float2(0.5, -0.5) + 0.5;
    return output;
}
