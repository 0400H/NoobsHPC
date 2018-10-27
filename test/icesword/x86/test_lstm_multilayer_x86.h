/* Copyright (c) 2016 NoobsDNN Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef NBDNN_TEST_ICESWORD_TEST_ICESWORD_MULTI_LSTM_X86_H
#define NBDNN_TEST_ICESWORD_TEST_ICESWORD_MULTI_LSTM_X86_H

#include "x86_test_common.h"

#include "icesword/funcs/lstm.h"
#include "icesword/funcs/impl/x86/x86_common.h"
#include "icesword/funcs/impl/x86/activation_functions.h"

using namespace noobsdnn::test;

class TestSaberMultiLSTMX86 : public Test {
public:
    TestSaberMultiLSTMX86() {}
    ~TestSaberMultiLSTMX86() {}

protected:
    virtual void setup() {}
    virtual void teardown() {}
};

#endif // NBDNN_TEST_ICESWORD_TEST_ICESWORD_MULTI_LSTM_X86_H
