set(OFX_SUPPORT_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB_RECURSE OFX_SUPPORT_HEADER_FILES "${OFX_SUPPORT_HEADER_DIR}/*.h")

add_subdirectory(Library)
  add_subdirectory(Plugins)
