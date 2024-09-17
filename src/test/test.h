/***********************************************
 * Author: chenxuan-1607772321@qq.com
 * change time:2024-04-15 14:59
 * description: this is a simple C++ testing framework
 * download or update:
 *    `wget https://gitee.com/chenxuan520/cpptest/raw/master/test.h -O test.h`
 ***********************************************/
#pragma once
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cpptest {

// base class
class _test_base {
public:
  bool result_ = true;
  bool is_async_ = false;
  std::string err_msg_ = "";
  std::string group_name_ = "";
  std::string test_name_ = "";
  static std::string regex_filt_;
  static std::vector<std::pair<std::string, _test_base *>> test_arr_;
  static int success_;
  static int fail_;

public:
  virtual void TestBody(){};
  _test_base() {}
  _test_base(const std::string &group_name, const std::string &test_name,
             bool is_async = false)
      : group_name_(group_name), test_name_(test_name), is_async_(is_async) {
    this->Init(group_name, test_name);
  }
  void Init(const std::string &group_name, const std::string &test_name) {
    group_name_ = group_name;
    test_name_ = test_name;
    test_arr_.push_back({group_name + " " + test_name_, this});
    success_++;
  }
};
int _test_base::success_ = 0;
int _test_base::fail_ = 0;
std::string _test_base::regex_filt_ = "";
std::vector<std::pair<std::string, _test_base *>> _test_base::test_arr_;

// for TEST_F
class Test : public _test_base {
public:
  virtual void SetUp() {}
};

// benchmark base
class _benchmark_base {
public:
  std::string group_name_ = "";
  std::string bench_name_ = "";
  static std::string regex_filt_;
  static std::vector<std::pair<std::string, _benchmark_base *>> bench_arr_;

public:
  void Init(const std::string &group_name, const std::string bench_name) {
    group_name_ = group_name;
    bench_name_ = bench_name;
    bench_arr_.push_back({group_name + " " + bench_name, this});
  }
  void Reset() {
    average_time_ = 0;
    last_time_ = 0;
    run_time_now_ = 0;
    run_time_ = 1;
  }
  int GetCycleTime() {
    // 计算方法参考 go test benchmark
    // https://www.bookstack.cn/read/GoExpertProgramming/chapter07-7.3.4-benchmark.md

    if (custom_run_time_ != 0) {
      return custom_run_time_;
    }

    if (run_time_now_ < run_time_) {
      return run_time_ + 1;
    }
    // 至少运行1s
    if ((run_time_now_ + 1) * average_time_ >=
            (long long)run_limit_second_ * 1e9 ||
        run_time_now_ >= 1e9) {
      return run_time_;
    }
    // 计算下一次运行次数,至少运行了一次,average_time_不为0
    int prediction_time = run_limit_second_ * 1e9 / average_time_;
    run_time_ = std::max(
        std::min(prediction_time + prediction_time / 5, 100 * run_time_),
        run_time_ + 1);
    return run_time_ + 1;
  }
  void CalcRunTime(int run_time) {}
  inline long long average_time() { return average_time_; }
  inline long long run_time() { return run_time_; }
  inline int custom_run_time() { return custom_run_time_; }
  inline void set_custom_run_time(int run_time) { custom_run_time_ = run_time; }
  inline void set_bench_msg(const std::string &msg) { bench_msg_ = " " + msg; }
  inline void set_run_limit_second(int run_limit_second) {
    run_limit_second_ = run_limit_second;
  }
  virtual void BenchBody() {}

protected:
  long long average_time_ = 0;
  long long last_time_ = 0;
  int run_time_now_ = 0;
  int run_time_ = 1;

protected:
  int custom_run_time_ = 0;
  int run_limit_second_ = 1;
  std::string bench_msg_ = "";
};
std::string _benchmark_base::regex_filt_ = "";
std::vector<std::pair<std::string, _benchmark_base *>>
    _benchmark_base::bench_arr_;

// std out or stderr
#define _TESTSTDOUT_(text) std::cout << text << std::endl;
#define _TESTSTDERR_(text) std::cerr << text << std::endl;

// print kinds of color
#define _TESTRED_(text) "\033[31m" << text << "\033[0m"
#define _TESTGREEN_(text) "\033[32m" << text << "\033[0m"
#define _TESTYELLOW_(text) "\033[33m" << text << "\033[0m"
#define _TESTBLUE_(text) "\033[34m" << text << "\033[0m"
#define _TESTCAR_(text) "\033[35m" << text << "\033[0m"
#define _TESTCYAN_(text) "\033[36m" << text << "\033[0m"

// setting macro
#define REGEX_FILT_TEST(test_regex)                                            \
  cpptest::_test_base::regex_filt_ = test_regex;                               \
  cpptest::_benchmark_base::regex_filt_ = test_regex;

// util macro
#define _CONNECTSTR_(...) #__VA_ARGS__
#define _CLASS_FAIL_(text)                                                     \
  std::ostringstream oss;                                                      \
  oss << text;                                                                 \
  this->err_msg_ = oss.str();                                                  \
  cpptest::_test_base::fail_++;                                                \
  cpptest::_test_base::success_--;                                             \
  this->result_ = false;
#define _FILE_LINE_MSG_ __FILE__ << ":" << __LINE__

// test name
#define _TEST_NAME_CREATE_(test_group, test_name) test_group##test_name##_create
#define _TEST_NAME_(test_group, test_name) test_group##test_name##_test
#define _TEST_TOOL_NAME_CREATE_(tool_name, create_name)                        \
  tool_name##create_name##_create
#define _TEST_INIT_NAME_CREATE_(init_name)                                     \
  _TEST_TOOL_NAME_CREATE_(init_, init_name)
#define _TEST_END_NAME_CREATE_(end_name) _TEST_TOOL_NAME_CREATE_(end_, end_name)
#define _TEST_WAITING_NAME_CREATE_(waiting_name)                               \
  _TEST_TOOL_NAME_CREATE_(wait_, waiting_name)
#define _TEST_GOROUTINE_NAME_CREATE_(go_name)                                  \
  _TEST_TOOL_NAME_CREATE_(go_, go_name)
#define _TEST_DEFER_NAME_CREATE_(defer_name)                                   \
  _TEST_TOOL_NAME_CREATE_(defer_, defer_name)
#define _TEST_INIT_NAME_(init_name) init_name##_init
#define _TEST_END_NAME_(end_name) end_name##_end

// test init function,run before all test example
#define INIT(init_name)                                                        \
  class _TEST_INIT_NAME_(init_name) {                                          \
  public:                                                                      \
    _TEST_INIT_NAME_(init_name)();                                             \
  };                                                                           \
  _TEST_INIT_NAME_(init_name) _TEST_INIT_NAME_CREATE_(init_name);              \
  _TEST_INIT_NAME_(init_name)::_TEST_INIT_NAME_(init_name)()

// test end function,run after all test example
#define END(end_name)                                                          \
  class _TEST_END_NAME_(end_name) {                                            \
  public:                                                                      \
    ~_TEST_END_NAME_(end_name)();                                              \
  };                                                                           \
  _TEST_END_NAME_(end_name) _TEST_END_NAME_CREATE_(end_name);                  \
  _TEST_END_NAME_(end_name)::~_TEST_END_NAME_(end_name)()

// defer func like Go
class _defer_ {
public:
  _defer_(std::function<void(void)> func_ptr) : func_ptr_(func_ptr) {}
  ~_defer_() { func_ptr_(); }
  std::function<void(void)> func_ptr_;
};
#define DEFER(func_ptr)                                                        \
  cpptest::_defer_ _TEST_DEFER_NAME_CREATE_(__LINE__)(func_ptr)
#define DEFER_DEFAULT                                                          \
  cpptest::_defer_ _TEST_DEFER_NAME_CREATE_(__LINE__)(nullptr);                \
  _TEST_DEFER_NAME_CREATE_(__LINE__).func_ptr_ = [&](void) -> void

// muti thread run func support,like go's goroutine
#define GO(func) std::thread(func).detach();
#define GO_JOIN(func)                                                          \
  auto _TEST_GOROUTINE_NAME_CREATE_(__LINE__) = std::thread(func);             \
  DEFER([&]() -> void {                                                        \
    if (_TEST_GOROUTINE_NAME_CREATE_(__LINE__).joinable()) {                   \
      _TEST_GOROUTINE_NAME_CREATE_(__LINE__).join();                           \
    }                                                                          \
  });
#define GO_WAIT(func) std::move(std::thread(func))

// like go wait group
class WaitGroup {
public:
  void Add(std::thread t) { thread_arr_.push_back(std::move(t)); }
  void Wait() {
    for (auto &t : thread_arr_)
      t.join();
  }

private:
  std::vector<std::thread> thread_arr_;
};

// calc cost time run func,return long long stand for,nano second
// 1s = 1000000000ns
#define COSTTIME(func)                                                         \
  [&]() -> long long {                                                         \
    auto start = std::chrono::high_resolution_clock::now();                    \
    func();                                                                    \
    auto end = std::chrono::high_resolution_clock::now();                      \
    auto duration =                                                            \
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);     \
    return duration.count();                                                   \
  }()

// bench run func,return cost time
#define BENCH_INDEPENDENT(func, run_time)                                      \
  [&]() -> long long {                                                         \
    if (run_time <= 0) {                                                       \
      return 0;                                                                \
    }                                                                          \
    auto start = std::chrono::high_resolution_clock::now();                    \
    for (int _test_i = 0; _test_i < run_time; ++_test_i) {                     \
      func();                                                                  \
    }                                                                          \
    auto end = std::chrono::high_resolution_clock::now();                      \
    auto duration =                                                            \
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);     \
    return duration.count() / run_time;                                        \
  }()

// bench function for users
#define BENCHMARK(bench_group, bench_name)                                     \
  class _TEST_NAME_(bench_group, bench_name)                                   \
      : public cpptest::_benchmark_base {                                      \
  public:                                                                      \
    _TEST_NAME_(bench_group, bench_name)() {                                   \
      this->Init(_CONNECTSTR_(bench_group), _CONNECTSTR_(bench_name));         \
    }                                                                          \
    void BenchBody();                                                          \
  };                                                                           \
  _TEST_NAME_(bench_group, bench_name)                                         \
  _TEST_NAME_CREATE_(bench_group, bench_name);                                 \
  void _TEST_NAME_(bench_group, bench_name)::BenchBody()

#define BENCHFUNC(func)                                                        \
  this->Reset();                                                               \
  for (int i = 0; i < this->GetCycleTime(); i++) {                             \
    this->last_time_ = COSTTIME(func);                                         \
    this->run_time_now_++;                                                     \
    this->average_time_ =                                                      \
        (this->average_time_ * i + this->last_time_) / (i + 1);                \
  }                                                                            \
  _TESTSTDOUT_(_TESTGREEN_("[Benchmark]:in ["                                  \
                           << _FILE_LINE_MSG_ << this->bench_msg_              \
                           << "]: Average Cost:" << this->average_time()       \
                           << " ns"                                            \
                           << " Run Time:"                                     \
                           << (this->custom_run_time() == 0                    \
                                   ? this->run_time()                          \
                                   : this->custom_run_time())));

// test function for users
#define TEST(test_group, test_name)                                            \
  class _TEST_NAME_(test_group, test_name) : public cpptest::_test_base {      \
  public:                                                                      \
    _TEST_NAME_(test_group, test_name)                                         \
    () : _test_base(_CONNECTSTR_(test_group), _CONNECTSTR_(test_name)) {}      \
    void TestBody();                                                           \
  };                                                                           \
  _TEST_NAME_(test_group, test_name)                                           \
  _TEST_NAME_CREATE_(test_group, test_name);                                   \
  void _TEST_NAME_(test_group, test_name)::TestBody()

#define TEST_ASYNC(test_group, test_name)                                      \
  class _TEST_NAME_(test_group, test_name) : public cpptest::_test_base {      \
  public:                                                                      \
    _TEST_NAME_(test_group, test_name)                                         \
    ()                                                                         \
        : _test_base(_CONNECTSTR_(test_group), _CONNECTSTR_(test_name),        \
                     true) {}                                                  \
    void TestBody();                                                           \
  };                                                                           \
  _TEST_NAME_(test_group, test_name)                                           \
  _TEST_NAME_CREATE_(test_group, test_name);                                   \
  void _TEST_NAME_(test_group, test_name)::TestBody()

#define TEST_F(test_class, test_name)                                          \
  class _TEST_NAME_(test_class, test_name) : public test_class {               \
  public:                                                                      \
    _TEST_NAME_(test_class, test_name)() {                                     \
      this->Init(_CONNECTSTR_(test_class), _CONNECTSTR_(test_name));           \
      this->SetUp();                                                           \
    }                                                                          \
    void TestBody();                                                           \
  };                                                                           \
  _TEST_NAME_(test_class, test_name)                                           \
  _TEST_NAME_CREATE_(test_class, test_name);                                   \
  void _TEST_NAME_(test_class, test_name)::TestBody()

// for use default name for create test example
#define TEST_DEFAULT TEST(TestDefault, __COUNTER__)

// some function for debug and judge
#define SKIP()                                                                 \
  _TESTSTDOUT_(_TESTYELLOW_("[SKIP]:in [" << _FILE_LINE_MSG_ << "] : "));      \
  return;
#define DEBUG(text)                                                            \
  _TESTSTDOUT_(                                                                \
      _TESTYELLOW_("[DEBUG]:in [" << _FILE_LINE_MSG_ << "] : " << text))
#define ERROR(text)                                                            \
  _TESTSTDERR_(_TESTCAR_("[ERROR]:in [" << _FILE_LINE_MSG_ << "] : " << text)) \
  _CLASS_FAIL_(text)
#define FATAL(text)                                                            \
  _TESTSTDERR_(_TESTRED_("[FATAL]:in [" << _FILE_LINE_MSG_ << "] : " << text)) \
  _CLASS_FAIL_(text)                                                           \
  return;
#define PANIC(text)                                                            \
  _TESTSTDERR_(                                                                \
      _TESTRED_("[PANIC]:in [" << _FILE_LINE_MSG_ << "] : " << text));         \
  exit(-1);
#define MUST_EQUAL(result, expect)                                             \
  if (result != expect) {                                                      \
    FATAL(result << " " << _CONNECTSTR_(!= expect));                           \
  }
#define MUST_TRUE(flag, err_msg)                                               \
  if (!(flag)) {                                                               \
    FATAL(err_msg);                                                            \
  }
#define EXPECT_EQ(result, expect)                                              \
  if (result != expect) {                                                      \
    ERROR(_CONNECTSTR_(result want get expect but get) << " " << result)       \
  }
#define EXPECT_TRUE(flag)                                                      \
  if (!(flag)) {                                                               \
    ERROR(_CONNECTSTR_(flag is false))                                         \
  }
#define ASSERT_EQ(result, expect) MUST_EQUAL(result, expect)
#define ASSERT_TRUE(flag) MUST_TRUE(flag, "")

// for gtest compatible
#define SUCCEED() SKIP()
#define FAIL() FATAL("")

// for argc message
static void (*__test_argc_funcpr__)(int argc, char *argv[]) = nullptr;
static void ArgcFunc(int argc, char **argv);
#define ARGC_FUNC                                                              \
  auto __test_temp_argc__ =                                                    \
      (cpptest::__test_argc_funcpr__ = cpptest::ArgcFunc);                     \
  void cpptest::ArgcFunc(int argc, char *argv[])

// the main function
#define RUN_ALL_TESTS                                                          \
  cpptest::_test_base base;                                                    \
  std::vector<std::pair<std::string, std::string>> fail_msg_arr;               \
  for (int i = 0; i < base.test_arr_.size(); i++) {                            \
    if (base.regex_filt_ != "") {                                              \
      std::regex pattern(base.regex_filt_);                                    \
      if (!std::regex_search(base.test_arr_[i].first, pattern)) {              \
        continue;                                                              \
      }                                                                        \
    }                                                                          \
    if (base.test_arr_[i].second->is_async_) {                                 \
      continue;                                                                \
    }                                                                          \
    _TESTSTDOUT_(_TESTCYAN_("[Runing]:" << base.test_arr_[i].first));          \
    auto start = std::chrono::high_resolution_clock::now();                    \
    base.test_arr_[i].second->TestBody();                                      \
    auto end = std::chrono::high_resolution_clock::now();                      \
    std::chrono::duration<double> duration = end - start;                      \
    if (base.test_arr_[i].second->result_) {                                   \
      _TESTSTDOUT_(_TESTGREEN_("[Result]:PASS"                                 \
                               << " Cost:" << duration.count() << "s")         \
                   << std::endl);                                              \
    } else {                                                                   \
      _TESTSTDOUT_(_TESTRED_("[Result]:Fail"                                   \
                             << " Cost:" << duration.count() << "s")           \
                   << std::endl);                                              \
      fail_msg_arr.push_back(                                                  \
          {base.test_arr_[i].first, base.test_arr_[i].second->err_msg_});      \
    }                                                                          \
  }                                                                            \
  if (base.regex_filt_ != "") {                                                \
    _TESTSTDOUT_(_TESTBLUE_("[Regex Filt]:" << base.regex_filt_) << std::endl) \
  }                                                                            \
  _TESTSTDOUT_(_TESTBLUE_("[Total Run]:" << base.success_ + base.fail_))       \
  _TESTSTDOUT_(_TESTBLUE_("[Success Run]:" << base.success_))                  \
  _TESTSTDERR_(_TESTBLUE_("[Fail Run]:" << base.fail_))                        \
  for (int i = 0; i < base.fail_; i++) {                                       \
    _TESTSTDERR_(_TESTRED_("[Fail Name]" << fail_msg_arr[i].first << " [Msg]:" \
                                         << fail_msg_arr[i].second));          \
  }                                                                            \
  _TESTSTDOUT_("") // for next line

// run all async test
#define RUN_ALL_ASYNC_TESTS                                                    \
  cpptest::_test_base base_sync;                                               \
  cpptest::WaitGroup wait_group;                                               \
  for (int i = 0; i < base_sync.test_arr_.size(); i++) {                       \
    if (base_sync.regex_filt_ != "") {                                         \
      std::regex pattern(base_sync.regex_filt_);                               \
      if (!std::regex_search(base_sync.test_arr_[i].first, pattern)) {         \
        continue;                                                              \
      }                                                                        \
    }                                                                          \
    if (!base_sync.test_arr_[i].second->is_async_) {                           \
      continue;                                                                \
    }                                                                          \
    wait_group.Add(std::thread([&base_sync, i]() -> void {                     \
      base_sync.test_arr_[i].second->TestBody();                               \
      std::string result = "[Async Test]" + base_sync.test_arr_[i].first;      \
      if (base_sync.test_arr_[i].second->result_) {                            \
        result += ":PASS";                                                     \
        _TESTSTDOUT_(_TESTGREEN_(result));                                     \
      } else {                                                                 \
        result += ":Fail" + base_sync.test_arr_[i].second->err_msg_;           \
        _TESTSTDOUT_(_TESTRED_(result));                                       \
      }                                                                        \
    }));                                                                       \
  }                                                                            \
  wait_group.Wait();

// run all bench test
#define RUN_ALL_BENCH                                                          \
  cpptest::_benchmark_base bench;                                              \
  if (bench.bench_arr_.size() != 0) {                                          \
    for (int i = 0; i < bench.bench_arr_.size(); i++) {                        \
      if (bench.regex_filt_ != "") {                                           \
        std::regex pattern(bench.regex_filt_);                                 \
        if (!std::regex_search(bench.bench_arr_[i].first, pattern)) {          \
          continue;                                                            \
        }                                                                      \
      }                                                                        \
      _TESTSTDOUT_(_TESTCYAN_(std::endl                                        \
                              << "[Runing]:" << bench.bench_arr_[i].first));   \
      bench.bench_arr_[i].second->BenchBody();                                 \
    }                                                                          \
    if (bench.regex_filt_ != "") {                                             \
      _TESTSTDOUT_(_TESTBLUE_(std::endl                                        \
                              << "[Regex Bench Filt]:" << bench.regex_filt_)   \
                   << std::endl)                                               \
    }                                                                          \
    _TESTSTDOUT_(_TESTBLUE_(std::endl                                          \
                            << "[Total Bench Run]:"                            \
                            << bench.bench_arr_.size()))                       \
  }

#define _RUN_TEST_MAIN_                                                        \
  int main(int argc, char *argv[]) {                                           \
    if (cpptest::__test_argc_funcpr__ != nullptr) {                            \
      cpptest::__test_argc_funcpr__(argc, argv);                               \
    }                                                                          \
    RUN_ALL_TESTS                                                              \
    RUN_ALL_ASYNC_TESTS                                                        \
    RUN_ALL_BENCH                                                              \
    return base.fail_;                                                         \
  }

}; // namespace cpptest

// default run auto
#ifndef TEST_CUSTOM_MAIN
_RUN_TEST_MAIN_
#endif
