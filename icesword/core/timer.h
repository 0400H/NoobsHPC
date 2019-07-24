/* Copyright (c) 2018 NoobsHPC Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef TIMER_H
#define TIMER_H

#pragma once

#include <list>

class Timer {

public:
    Timer() {}

    ~Timer() {}

    enum time_mode {lasted = 0, min = 1, max = 2, avg = 3, total = 4};

    void clear() {
        time_list.clear();
    }

    void start() {
        tstart = std::chrono::system_clock::now();
    }

    void stop() {
        tstop = std::chrono::system_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::microseconds>(tstop - tstart);
        double elapse_ms = 1000.f * double(ts.count()) * std::chrono::microseconds::period::num / \
                           std::chrono::microseconds::period::den;
        time_list.push_back(elapse_ms);
        if (elapse_ms > time_max) { time_max = elapse_ms; }
        if (elapse_ms < time_min || time_list.size() == 1) { time_min = elapse_ms; }
    }

    double get_time_ms(time_mode mode = Timer::avg) {
        if (time_list.size() == 0) {
            return 0.f;
        }
        switch (mode) {
            case Timer::lasted :
                return time_lasted;
            case Timer::min :
                return time_min;
            case Timer::max :
                return time_max;
            case Timer::avg :
                time_sum = 0;
                for (auto i : time_list){
                    time_sum += i;
                }
                return time_sum / time_list.size();
            case Timer::total :
                time_sum = 0;
                for (auto i : time_list){
                    time_sum += i;
                }
                return time_sum;
            default :
                return -1;
        }
    }

    const std::list<double> get_time_stat() {
        return time_list;
    }

private:
    std::list<double> time_list;
    double time_max = 0, time_min = 0, time_lasted = 0, time_sum = 0;
    std::chrono::time_point<std::chrono::system_clock> tstart;
    std::chrono::time_point<std::chrono::system_clock> tstop;

};

#endif // TIMER_H
