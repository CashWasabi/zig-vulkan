const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("GLFW/glfw3.h");
    @cInclude("vulkan/vulkan.h");
});
const C = std.builtin.CallingConvention.c;
const std = @import("std");

pub fn main() !void {
    if (c.glfwInit() == c.GLFW_FALSE) @panic("glwfInit failed!");
    defer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    const window: ?*c.GLFWwindow = c.glfwCreateWindow(800, 600, "Vulkan window", null, null);
    defer c.glfwDestroyWindow(window);

    var extension_count: u32 = 0;
    if (c.vkEnumerateInstanceExtensionProperties(null, &extension_count, null) != c.VK_SUCCESS) {
        @panic("vkEnumerateInstanceExtensionProperties failed!");
    }

    std.log.debug("{} extensions supported.", .{extension_count});
    while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE) {
        c.glfwPollEvents();
    }
}

// int main() {
//     glfwInit();
//
//     glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
//     GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
//
//     uint32_t extensionCount = 0;
//     vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
//
//     std::cout << extensionCount << " extensions supported\n";
//
//     glm::mat4 matrix;
//     glm::vec4 vec;
//     auto test = matrix * vec;
//
//     while(!glfwWindowShouldClose(window)) {
//         glfwPollEvents();
//     }
//
//     glfwDestroyWindow(window);
//
//     glfwTerminate();
//
//     return 0;
// }
