glslc -fshader-stage=vert ./src/vulkan_tutorial/drawing_triangle/shaders/vert.glsl -o ./src/vulkan_tutorial/drawing_triangle/spv/vert.spv
glslc -fshader-stage=frag ./src/vulkan_tutorial/drawing_triangle/shaders/frag.glsl -o ./src/vulkan_tutorial/drawing_triangle/spv/frag.spv

glslc -fshader-stage=vert ./src/vulkan_tutorial/vertex_buffers/shaders/vert.glsl -o ./src/vulkan_tutorial/vertex_buffers/spv/vert.spv
glslc -fshader-stage=frag ./src/vulkan_tutorial/vertex_buffers/shaders/frag.glsl -o ./src/vulkan_tutorial/vertex_buffers/spv/frag.spv
