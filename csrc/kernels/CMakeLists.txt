function(add_deep_ep_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD_REQUIRED ON
            CXX_STANDARD 20
    )
    target_link_libraries(${target_name} PUBLIC CUDA::cudart)
endfunction()

add_subdirectory(comm)
add_subdirectory(driver)

set(RUNTIME_CUDA_LIBRARIES runtime_comm runtime_driver CACHE INTERNAL "Runtime kernels")
