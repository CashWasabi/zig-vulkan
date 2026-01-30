glslc -fshader-stage=vert ./src/vulkan_tutorial/01_drawing_triangle/shaders/vert.glsl -o ./src/vulkan_tutorial/01_drawing_triangle/spv/vert.spv
glslc -fshader-stage=frag ./src/vulkan_tutorial/01_drawing_triangle/shaders/frag.glsl -o ./src/vulkan_tutorial/01_drawing_triangle/spv/frag.spv

glslc -fshader-stage=vert ./src/vulkan_tutorial/02_vertex_buffers/shaders/vert.glsl -o ./src/vulkan_tutorial/02_vertex_buffers/spv/vert.spv
glslc -fshader-stage=frag ./src/vulkan_tutorial/02_vertex_buffers/shaders/frag.glsl -o ./src/vulkan_tutorial/02_vertex_buffers/spv/frag.spv

glslc -fshader-stage=vert ./src/vulkan_tutorial/03_uniform_buffers/shaders/vert.glsl -o ./src/vulkan_tutorial/03_uniform_buffers/spv/vert.spv
glslc -fshader-stage=frag ./src/vulkan_tutorial/03_uniform_buffers/shaders/frag.glsl -o ./src/vulkan_tutorial/03_uniform_buffers/spv/frag.spv

glslc -fshader-stage=vert ./src/vulkan_tutorial/04_texture_mapping/shaders/vert.glsl -o ./src/vulkan_tutorial/04_texture_mapping/spv/vert.spv
glslc -fshader-stage=frag ./src/vulkan_tutorial/04_texture_mapping/shaders/frag.glsl -o ./src/vulkan_tutorial/04_texture_mapping/spv/frag.spv
