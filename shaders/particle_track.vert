#version 330 core
layout(location = 0) in vec3 aPosition;  
layout(location = 1) in float aEnergy;   

uniform mat4 uMVP;      

out float vEnergy;     

void main()
{
    vEnergy    = aEnergy;
    gl_Position = uMVP * vec4(aPosition, 1.0);
}
