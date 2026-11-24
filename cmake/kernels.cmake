# Kernels are OpenCL C, compiled ahead of time by ocloc into native binaries (ADR-0003), one set per chip in
# ENGINE_AOT_DEVICES, each in a directory named for the chip's IP version, which the core matches to each card's. ocloc
# writes each module's SPIR-V beside its binary; the core builds a card's modules from it when no directory matches.
# The source is built with -cl-std=CL3.0: without it ocloc compiles OpenCL C 1.2. Divides and sqrt are rounded
# correctly, as ggml's CPU code rounds them, so a quantizer's amax / 127 gives ggml's bytes (ADR-0052).

find_program(ENGINE_OCLOC ocloc)
if(NOT ENGINE_OCLOC)
  message(FATAL_ERROR "ocloc not found: install intel-ocloc, pinned in docs/adr/0002-pin-the-toolchain.md")
endif()
set(ENGINE_AOT_DEVICES "bmg-g21;bmg-g31" CACHE STRING "ocloc -device values, one binary set each (ADR-0002, ADR-0057)")
# Chips that take the 8-lane DPAS kernels (kernels/include/dg2.h): their modules build with -DENGINE_DG2, which
# selects the Alchemist paths. The Battlemage devices' commands are unchanged.
set(ENGINE_DG2_DEVICES "acm-g10" CACHE STRING "ocloc -device values built with -DENGINE_DG2")
# The device whose SPIR-V is the load-time fallback. It must be the DG2 set where one is built: the Battlemage
# SPIR-V carries the 2D block reads and the 16-lane DPAS an Alchemist card cannot run. Empty: the first AOT device.
set(ENGINE_KERNEL_SPIRV_DEVICE "" CACHE STRING "ocloc -device whose SPIR-V the core compiles at load")
set(ENGINE_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "")
file(MAKE_DIRECTORY ${ENGINE_KERNEL_DIR})

# Each device's IP version, as `ocloc ids` gives it and Level Zero reports it: "20.1.0" for bmg-g21.
set(ENGINE_KERNEL_IPS "")
foreach(device ${ENGINE_AOT_DEVICES})
  execute_process(COMMAND ${ENGINE_OCLOC} ids ${device} OUTPUT_VARIABLE ids RESULT_VARIABLE rc)
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" ip "${ids}")
  if(NOT rc EQUAL 0 OR NOT ip)
    message(FATAL_ERROR "ocloc ids ${device} gave no IP version")
  endif()
  list(APPEND ENGINE_KERNEL_IPS ${ip})
  file(MAKE_DIRECTORY ${ENGINE_KERNEL_DIR}/${ip})
endforeach()
if(ENGINE_KERNEL_SPIRV_DEVICE)
  list(FIND ENGINE_AOT_DEVICES "${ENGINE_KERNEL_SPIRV_DEVICE}" spirv_index)
  if(spirv_index LESS 0)
    message(FATAL_ERROR "ENGINE_KERNEL_SPIRV_DEVICE=${ENGINE_KERNEL_SPIRV_DEVICE} is not in ENGINE_AOT_DEVICES")
  endif()
  list(GET ENGINE_KERNEL_IPS ${spirv_index} ENGINE_KERNEL_SPIRV_IP)
else()
  list(GET ENGINE_KERNEL_IPS 0 ENGINE_KERNEL_SPIRV_IP)
endif()
set(ENGINE_KERNEL_SPIRV_DIR "${ENGINE_KERNEL_DIR}/${ENGINE_KERNEL_SPIRV_IP}" CACHE INTERNAL "" FORCE)
# Marks the fallback set as the portable one: a DG2 card refuses a fallback built for Battlemage (core/ze/ze.cpp).
file(REMOVE ${ENGINE_KERNEL_SPIRV_DIR}/.dg2)
if(ENGINE_KERNEL_SPIRV_DEVICE IN_LIST ENGINE_DG2_DEVICES)
  file(WRITE ${ENGINE_KERNEL_SPIRV_DIR}/.dg2 "")
endif()

# engine_add_kernel_module(<name> SOURCE <file.cl> [OPTIONS <build options>...])
# Writes <ip>/<name>.bin and <ip>/<name>.spv for each device, and <name>.options: the -cl- options a build from the
# SPIR-V needs. A variant is another module of the same source with other -D options. Every header under kernels/ and
# shared/ is a dependency, and both directories are on the include path.
function(engine_add_kernel_module name)
  cmake_parse_arguments(K "SKIP_DG2" "SOURCE" "OPTIONS" ${ARGN})
  set(src ${CMAKE_SOURCE_DIR}/${K_SOURCE})
  file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/kernels/*.h ${CMAKE_SOURCE_DIR}/shared/*.h)
  list(JOIN K_OPTIONS " " opts)
  set(native "-cl-fp32-correctly-rounded-divide-sqrt")
  foreach(o ${K_OPTIONS})
    if(o MATCHES "^-cl-")
      string(APPEND native " ${o}")
    endif()
  endforeach()
  file(WRITE ${ENGINE_KERNEL_DIR}/${name}.options "${native}")
  set(outputs "")
  foreach(device ip IN ZIP_LISTS ENGINE_AOT_DEVICES ENGINE_KERNEL_IPS)
    if(K_SKIP_DG2 AND device IN_LIST ENGINE_DG2_DEVICES)
      continue()
    endif()
    set(out ${ENGINE_KERNEL_DIR}/${ip})
    set(dev_opts "${opts}")
    if(device IN_LIST ENGINE_DG2_DEVICES)
      string(APPEND dev_opts " -DENGINE_DG2")
    endif()
    add_custom_command(
      OUTPUT ${out}/${name}.bin ${out}/${name}.spv
      COMMAND ${ENGINE_OCLOC} compile -q -file ${src} -device ${device}
              -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -I${CMAKE_SOURCE_DIR}/kernels/include -I${CMAKE_SOURCE_DIR}/shared ${dev_opts}"
              -out_dir ${out} -output ${name} -output_no_suffix
      DEPENDS ${src} ${headers}
      COMMENT "ocloc ${K_SOURCE} -> ${ip}/${name}.bin (${device})"
      VERBATIM)
    list(APPEND outputs ${out}/${name}.bin)
  endforeach()
  add_custom_target(kernel_${name} ALL DEPENDS ${outputs})
  set_property(GLOBAL APPEND PROPERTY ENGINE_KERNEL_TARGETS kernel_${name})
endfunction()
