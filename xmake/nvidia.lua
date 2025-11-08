target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- 禁用自动标志检查
    set_policy("check.auto_ignore_flags", false)
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-dc", "-Xcompiler", "-fPIC")  -- -dc: 设备代码分离编译
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
        add_culdflags("-Xcompiler", "-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")
    set_policy("build.cuda.devlink", true)
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")

    -- 禁用自动标志检查
    set_policy("check.auto_ignore_flags", false)
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-dc", "-Xcompiler", "-fPIC")  -- -dc: 设备代码分离编译
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
        add_culdflags("-Xcompiler", "-fPIC")
    end


    add_files("../src/ops/*/nvidia/**.cu")
    add_cugencodes("native")
    set_policy("build.cuda.devlink", true)
    on_install(function (target) end)
target_end()