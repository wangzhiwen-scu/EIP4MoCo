import matlab.engine
 
# 连接到 MATLAB engine
eng = matlab.engine.start_matlab()
 
# 运行 Matlab 函数或脚本
result = eng.eval('demo.m')
 
# 关闭与 MATLAB engine 的连接
eng.quit()


# $ python setup.py install --prefix="/local/work/matlab20aPy36" build --build-base="/home/wzw/miniconda3/envs/torch/bin" install --prefix="/home/wzw/miniconda3/envs/torch/"