
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_mpi_test.hpp>


int main(int argc, char * argv[]) {
    int ret = toast::test::mpi_runner(argc, argv);
    return ret;
}
