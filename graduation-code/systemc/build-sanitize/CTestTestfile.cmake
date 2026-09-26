# CMake generated Testfile for 
# Source directory: /Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc
# Build directory: /Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/build-sanitize
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(atu_hpu_demo "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/build-sanitize/atu_hpu_demo")
set_tests_properties(atu_hpu_demo PROPERTIES  _BACKTRACE_TRIPLES "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/CMakeLists.txt;66;add_test;/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/CMakeLists.txt;0;")
add_test(command_codec_tests "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/build-sanitize/command_codec_tests" "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/../fixtures/command_schema_v2")
set_tests_properties(command_codec_tests PROPERTIES  _BACKTRACE_TRIPLES "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/CMakeLists.txt;67;add_test;/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/CMakeLists.txt;0;")
add_test(fp64_reference_tests "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/build-sanitize/fp64_reference_tests" "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/reference/fixtures/fp64_reference_v1.json" "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/../fixtures/reference_tree/tree_policy_cases.json")
set_tests_properties(fp64_reference_tests PROPERTIES  _BACKTRACE_TRIPLES "/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/CMakeLists.txt;71;add_test;/Users/huangyuan/qcjjsyx/graduation/Graduation-Project/graduation-code/systemc/CMakeLists.txt;0;")
