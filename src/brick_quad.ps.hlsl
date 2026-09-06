struct ps_input
{
    nointerpolation uint entry_face : TEXCOORD0;
    float2               face_uv : TEXCOORD1;
};

uint main(ps_input const input) : SV_Target0
{
    // Reserve zero for pixels not covered by a brick face.
    // Bits 0..23 hold the brick coordinates; bits 24..26 hold the exposed face.
    // Mark a one-pixel border where raster coverage and reconstructed rays may disagree.
    float2 const edge = min(input.face_uv, 1.0 - input.face_uv);
    uint const   border = any(edge <= fwidth(input.face_uv)) ? 0x80000000u : 0u;
    return (input.entry_face + 1u) | border;
}
