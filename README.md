# pinkie

`pinkie` project with C++14

# Install
```sh
git clone --recursive http://ai-dev/UII/_git/pinkie
cd pinkie
mkdir build && cd build
cmake ../pinkie
make -j8
cd ..
pip install -e .
```

# Note

It need to add pyd to python search directory when use C++ interface pyd files, so we insert 'pinkie/lib/Release' at `pinkie's __init__.py`.

Please use 'pip install -e .' and 'import pinkie' first to use pinkie python.

You can change `pinkie's __init__.py` to 'Debug' if you want run with debug interface.

# torch
add torch dir to cmake prefix path
```python
import os
import torch
print(os.path.dirname(torch.__file__))
print(sys.executable)
```
add it to cmake config args
```shell
-DCMAKE_PREFIX_PATH=${torch_dir}
-DPYTHON_EXECUTABLE:FILEPATH=${python_path}
```

**please make sure pytorch install success**