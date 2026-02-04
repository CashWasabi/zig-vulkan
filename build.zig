const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const options = b.addOptions();
    options.addOption(
        bool,
        "enable_validation_layers",
        b.option(bool, "enable-validation-layers", "Enable validation layers") orelse false,
    );

    const obj_mod = b.dependency("obj", .{ .target = target, .optimize = optimize }).module("obj");

    const root_module = b.addModule("zig_vulkan", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    root_module.addOptions("build_options", options);
    root_module.addImport("obj", obj_mod);

    const exe = b.addExecutable(.{
        .name = "zig_vulkan",
        .root_module = root_module,
    });

    // include third_party libs
    exe.addIncludePath(b.path("third_party")); // link the library

    // include vulkan
    exe.linkSystemLibrary("vulkan"); // link the library

    // Include GLFW
    exe.linkSystemLibrary("glfw"); // link the library

    // linux
    exe.linkSystemLibrary("GL");
    exe.linkSystemLibrary("X11");
    exe.linkSystemLibrary("Xrandr");
    exe.linkSystemLibrary("Xi");
    exe.linkSystemLibrary("dl");
    exe.linkSystemLibrary("pthread");
    exe.linkSystemLibrary("m");

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}
