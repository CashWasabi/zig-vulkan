const builtin = @import("builtin");
// const build_options = @import("build_options");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("signal.h");
    @cInclude("GLFW/glfw3.h");
    // @cInclude("vulkan/vulkan.h");
});
const C = std.builtin.CallingConvention.c;
const std = @import("std");

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const ENABLE_VALIDATION_LAYERS: bool = builtin.mode == .Debug;
const VALIDATION_LAYERS = [_][:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

var g_should_exit: bool = false;

const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
};

fn handle_sigint(sig: c_int) callconv(C) void {
    _ = sig;
    g_should_exit = true;
}

pub fn main() !void {
    if (c.signal(c.SIGINT, handle_sigint) == c.SIG_ERR) {
        @panic("Failed to register SIGINT handler");
    }

    var dbga: std.heap.DebugAllocator(.{}) = .init;
    defer if (dbga.deinit() == .leak) @panic("Debug allocator has leaked meory!");
    const allocator = dbga.allocator();

    if (c.glfwInit() == c.GLFW_FALSE) @panic("glwfInit failed!");
    defer c.glfwTerminate();

    // ==============
    // CREATE WINDOW
    // ==============
    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_FALSE);
    const window: ?*c.GLFWwindow = c.glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", null, null);
    defer c.glfwDestroyWindow(window);

    // ========================================
    // CHECK IF WE HAVE ANY EXTENSION FEATURES
    // ========================================
    var extension_count: u32 = 0;
    if (c.vkEnumerateInstanceExtensionProperties(null, &extension_count, null) != c.VK_SUCCESS) {
        @panic("vkEnumerateInstanceExtensionProperties failed!");
    }

    // =======================
    // CREATE VULKAN INSTANCE
    // =======================
    if (ENABLE_VALIDATION_LAYERS and try checkValidationLayerSupport(allocator) == false) {
        @panic("Validation Layers not available!");
    }

    var app_info: c.VkApplicationInfo = .{};
    app_info.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Hello Triangle";
    app_info.applicationVersion = c.VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = c.VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = c.VK_API_VERSION_1_0;

    var create_info: c.VkInstanceCreateInfo = .{};
    create_info.sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    const extensions = try getRequiredExtensions(allocator);
    defer allocator.free(extensions);

    create_info.enabledExtensionCount = @intCast(extensions.len);
    create_info.ppEnabledExtensionNames = extensions.ptr;

    // SETUP VALIDATION LAYERS (OPTIONAL)
    var debug_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
    const validation_layers_slice = try allocator.alloc([*c]const u8, VALIDATION_LAYERS.len);
    defer allocator.free(validation_layers_slice);
    if (ENABLE_VALIDATION_LAYERS) {
        create_info.enabledLayerCount = @intCast(VALIDATION_LAYERS.len);
        for (0..VALIDATION_LAYERS.len) |i| validation_layers_slice[i] = VALIDATION_LAYERS[i].ptr;
        create_info.ppEnabledLayerNames = validation_layers_slice.ptr;

        populateDebugMessengerCreateInfo(&debug_create_info);
        create_info.pNext = &debug_create_info;
    } else {
        create_info.enabledLayerCount = 0;
        create_info.pNext = null;
    }

    var instance: c.VkInstance = undefined;
    // NOTE: the second parameter here is a custom memoru allocator callback!
    if (c.vkCreateInstance(&create_info, null, &instance) != c.VK_SUCCESS) {
        @panic("vkCreateInstance failed!");
    }
    defer c.vkDestroyInstance(instance, null);

    // =======================
    // SETUP DEBUG MESSENGER
    // =======================
    var debug_messenger: c.VkDebugUtilsMessengerEXT = undefined;
    var debug_messenger_create_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{};
    if (ENABLE_VALIDATION_LAYERS) {
        populateDebugMessengerCreateInfo(&debug_messenger_create_info);

        if (createDebugUtilsMessengerEXT(instance, &debug_messenger_create_info, null, &debug_messenger) != c.VK_SUCCESS) {
            @panic("failed to set up debug messenger!");
        }
    }
    defer destroyDebugUtilsMessengerEXT(instance, debug_messenger, null);

    // =====================
    // PICK PHYSICAL DEVICE
    // =====================
    var physical_device: c.VkPhysicalDevice = null;
    var device_count: u32 = 0;
    if (c.vkEnumeratePhysicalDevices(instance, &device_count, null) != c.VK_SUCCESS) {
        @panic("failed calling vkEnumeratePhysicalDevices while looking for devices count!");
    }
    if (device_count == 0) {
        @panic("failed to find GPUs with Vulkan support!");
    }
    const devices = try allocator.alloc(c.VkPhysicalDevice, device_count);
    defer allocator.free(devices);
    if (c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr) != c.VK_SUCCESS) {
        @panic("failed calling vkEnumeratePhysicalDevices while getting devices!");
    }
    for (devices) |device| {
        if (try isDeviceSuitable(allocator, device)) {
            physical_device = device;
            break;
        }
    }
    if (physical_device == null) {
        @panic("failed to find a suitable GPU!");
    }

    // ====================
    // PICK LOGICAL DEVICE
    // ====================

    // ==============
    // RUN MAIN LOOP
    // ==============
    // NOTE: we use sigint as long as we don't have a real window! Without a window we can't close correctly
    while (!g_should_exit) {
        // while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE or c.glfwGetKey(window, c.GLFW_KEY_ESCAPE) != c.GLFW_PRESS) {
        c.glfwPollEvents();
    }
}

pub fn checkValidationLayerSupport(allocator: std.mem.Allocator) !bool {
    var layer_count: u32 = 0;
    if (c.vkEnumerateInstanceLayerProperties(&layer_count, null) != c.VK_SUCCESS) {
        @panic("vkEnumerateInstanceLayerProperties in check_validation_layer_support failed!");
    }

    const available_layers = try allocator.alloc(c.VkLayerProperties, layer_count);
    defer allocator.free(available_layers);

    if (c.vkEnumerateInstanceLayerProperties(&layer_count, available_layers.ptr) != c.VK_SUCCESS) {
        @panic("vkEnumerateInstanceLayerProperties in check_validation_layer_support failed!");
    }

    for (VALIDATION_LAYERS) |layer_name| {
        var layer_found: bool = false;
        for (available_layers) |layer_properties| {
            var slices_eql: bool = true;
            for (0..layer_name.len) |i| {
                if (layer_name[i] != layer_properties.layerName[i]) {
                    slices_eql = false;
                    break;
                }
            }

            if (slices_eql) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) return false;
    }

    return true;
}

// TODO: do I need to do callconc(C) here?
pub fn getRequiredExtensions(allocator: std.mem.Allocator) ![][*c]const u8 {
    // NOTE: since vulkan is platform agnostic we use a glfw helper here.
    var glfw_extension_count: u32 = 0;
    const glfw_extensions = c.glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    if (ENABLE_VALIDATION_LAYERS) {
        const extensions = try allocator.alloc([*c]const u8, glfw_extension_count + 1);
        for (0..glfw_extension_count) |i| extensions[i] = glfw_extensions[i];
        extensions[extensions.len - 1] = c.VK_EXT_DEBUG_UTILS_EXTENSION_NAME.ptr;
        return extensions;
    } else {
        const extensions = try allocator.alloc([*c]const u8, glfw_extension_count);
        for (0..glfw_extension_count) |i| extensions[i] = glfw_extensions[i];
        return extensions;
    }
}

pub fn populateDebugMessengerCreateInfo(create_info: *c.VkDebugUtilsMessengerCreateInfoEXT) void {
    create_info.sType = c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debugCallback;
}

pub fn debugCallback(
    // VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    messageSeverity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
    // VkDebugUtilsMessageTypeFlagsEXT messageType,
    messageType: c.VkDebugUtilsMessageTypeFlagsEXT,
    // const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    pCallbackData: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
    // void* pUserData
    pUserData: ?*anyopaque,
    //) {
) callconv(C) u32 {
    _ = messageSeverity;
    _ = messageType;
    _ = pUserData;

    // std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    std.debug.print("validation layer: {s}\n", .{pCallbackData.*.pMessage});

    return c.VK_FALSE;
}

// NOTE: some functions need to be loaded first and then be casted into the right type!
//       this is (as in opengl) a thing for extensions. They need to be loaded at runtime.
pub fn createDebugUtilsMessengerEXT(
    instance: c.VkInstance,
    p_create_info: *const c.VkDebugUtilsMessengerCreateInfoEXT,
    p_allocator: ?*const c.VkAllocationCallbacks,
    p_debug_messenger: *c.VkDebugUtilsMessengerEXT,
) c.VkResult {
    const opt_func = c.vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (opt_func) |func| {
        const vkCreateDebugUtilsMessengerEXT_fn: c.PFN_vkCreateDebugUtilsMessengerEXT = @ptrCast(func);
        return vkCreateDebugUtilsMessengerEXT_fn.?(instance, p_create_info, p_allocator, p_debug_messenger);
    } else {
        return c.VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

pub fn destroyDebugUtilsMessengerEXT(
    instance: c.VkInstance,
    debug_messenger: c.VkDebugUtilsMessengerEXT,
    p_allocator: ?*const c.VkAllocationCallbacks,
) void {
    const opt_func = c.vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (opt_func) |func| {
        const vkDestroyDebugUtilsMessengerEXT_fn: c.PFN_vkDestroyDebugUtilsMessengerEXT = @ptrCast(func);
        vkDestroyDebugUtilsMessengerEXT_fn.?(instance, debug_messenger, p_allocator);
    }
}

pub fn isDeviceSuitable(allocator: std.mem.Allocator, device: c.VkPhysicalDevice) !bool {
    const indices: QueueFamilyIndices = try findQueueFamilies(allocator, device);
    return indices.graphics_family != null;
}

pub fn findQueueFamilies(allocator: std.mem.Allocator, device: c.VkPhysicalDevice) !QueueFamilyIndices {
    var queue_family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);
    const queue_families = try allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

    var indices: QueueFamilyIndices = .{};

    for (queue_families, 0..) |queue_family, i| {
        if ((queue_family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0) {
            indices.graphics_family = @intCast(i);
            break;
        }
    }

    return indices;
}
