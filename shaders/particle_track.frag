#version 330 core

in  float vEnergy;
out vec4  fragColor;

uniform float uMaxEnergy;

void main()
{
    float t     = clamp(vEnergy / max(uMaxEnergy, 1e-5), 0.0, 1.0);
    float alpha = mix(0.25, 1.0, t);
    fragColor   = vec4(1.0, 1.0, 1.0, alpha);
}