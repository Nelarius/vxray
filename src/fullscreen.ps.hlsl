struct ps_input
{
    float2 uv : TEXCOORD0;
};

float4 main(ps_input input) : SV_Target0 { return float4(input.uv, 0.0, 1.0); }
