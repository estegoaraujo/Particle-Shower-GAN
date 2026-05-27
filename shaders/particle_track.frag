#version 330 core
/**
 * particle_track.frag
 * -------------------
 * Fragment shader for particle tracks.
 *
 * Aesthetic: strict monochrome — pure white (#FFFFFF) on off-black.
 * In Phase 6 we will add energy-based alpha falloff so low-energy
 * secondaries appear slightly dimmer, preserving the minimalist look
 * while conveying physical depth.
 */

in  float vEnergy;
out vec4  fragColor;

uniform float uMaxEnergy;  ///< Primary particle energy [GeV] for normalisation

void main()
{
    // Normalise energy to [0,1] range (clamped)
    float t = clamp(vEnergy / max(uMaxEnergy, 1e-5), 0.0, 1.0);

    // For now: pure white. Phase 6 will encode energy as alpha.
    // fragColor = vec4(1.0, 1.0, 1.0, mix(0.3, 1.0, t));
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);  // Snow white #FFFFFF
}
