if(NOT DEFINED GIT_EXECUTABLE)
    message(FATAL_ERROR "GIT_EXECUTABLE is required")
endif()

if(NOT DEFINED PATCH_FILE)
    message(FATAL_ERROR "PATCH_FILE is required")
endif()

if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE apply_check_result
    ERROR_VARIABLE apply_check_error)

if(apply_check_result EQUAL 0)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply "${PATCH_FILE}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE apply_result
        ERROR_VARIABLE apply_error)
    if(NOT apply_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${PATCH_FILE}:\n${apply_error}")
    endif()
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE reverse_check_result
    ERROR_VARIABLE reverse_check_error)

if(reverse_check_result EQUAL 0)
    message(STATUS "Patch already applied: ${PATCH_FILE}")
    return()
endif()

message(
    FATAL_ERROR
        "Patch cannot be applied and is not already applied: ${PATCH_FILE}\n"
        "Apply check failed:\n${apply_check_error}"
        "Reverse check failed:\n${reverse_check_error}")
