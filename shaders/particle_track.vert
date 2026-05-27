#version 330 core
/**
 * particle_track.vert
 * -------------------
 * Vertex shader for rendering particle track line-strips.
 *
 * Input:  3D world-space position of each track point.
 * Output: Clip-space position passed to the fragment shader.
 *
 * The model-view-projection (MVP) matrix is supplied as a uniform.
 * In Phase 5 we will bind a camera that orbits the shower volume.
 */

layout(location = 0) in vec3 aPosition;  ///< Track point in world space [cm]
layout(location = 1) in float aEnergy;   ///< Energy at this track point [GeV]
                                         ///<   (used later for brightness encoding)

uniform mat4 uMVP;       ///< Model-View-Projection matrix

out float vEnergy;       ///< Pass energy to fragment shader

void main()
{
    vEnergy    = aEnergy;
    gl_Position = uMVP * vec4(aPosition, 1.0);
}
