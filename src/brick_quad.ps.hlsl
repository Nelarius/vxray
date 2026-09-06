struct ps_input
{
    nointerpolation uint packed_brick : TEXCOORD0;
};

uint main(ps_input const input) : SV_Target0
{
    // Reserve zero for pixels not covered by a brick face.
    return input.packed_brick + 1u;
}
