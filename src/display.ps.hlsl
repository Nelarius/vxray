struct ps_input
{
    float2 uv : TEXCOORD0;
};

Texture2D<float4> display_texture : register(t0, space2);
SamplerState      display_sampler : register(s0, space2);

float4 main(ps_input const input) : SV_Target0
{
    return display_texture.SampleLevel(display_sampler, input.uv, 0.0);
}
