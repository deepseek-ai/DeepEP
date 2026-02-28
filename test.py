import ctypes
import os
import sys

def load_shared_library(lib_path):
    """
    加载动态链接库并返回库对象
    
    参数:
        lib_path (str): 动态链接库的路径
        
    返回:
        ctypes.CDLL: 加载的库对象
        
    异常:
        OSError: 如果库无法加载
    """
    # 检查文件是否存在
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"库文件不存在: {lib_path}")
    
    # 检查文件是否可读
    if not os.access(lib_path, os.R_OK):
        raise PermissionError(f"没有读取权限: {lib_path}")
    
    try:
        # 尝试加载库
        lib = ctypes.CDLL(lib_path)
        print(f"成功加载库: {lib_path}")
        return lib
    except OSError as e:
        print(f"加载库失败: {e}")
        print("可能的原因:")
        print("1. 文件不是有效的动态链接库(ET_DYN或ET_EXEC)")
        print("2. 缺少依赖库")
        print("3. 架构不匹配(如x86_64库在aarch64系统上)")
        print("4. 权限问题")
        raise

def check_library_dependencies(lib_path):
    """
    使用ldd检查库的依赖项
    
    参数:
        lib_path (str): 动态链接库的路径
    """
    print("\n检查库依赖项:")
    try:
        result = os.system(f"ldd {lib_path}")
        if result != 0:
            print("ldd命令执行失败或发现依赖问题")
    except Exception as e:
        print(f"执行ldd时出错: {e}")

def get_library_info(lib_path):
    """
    获取库的基本信息
    
    参数:
        lib_path (str): 动态链接库的路径
    """
    print("\n库文件信息:")
    try:
        result = os.popen(f"file {lib_path}").read()
        print(result.strip())
    except Exception as e:
        print(f"获取文件信息时出错: {e}")

if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) < 2:
        print("用法: python dlopen_example.py <库文件路径>")
        sys.exit(1)
    
    lib_path = sys.argv[1]
    
    # 获取库信息
    get_library_info(lib_path)
    
    # 检查依赖项
    check_library_dependencies(lib_path)
    
    # 尝试加载库
    try:
        lib = load_shared_library(lib_path)
        
        # 列出库中可用的函数
        print("\n库中可用的函数:")
        for func_name in dir(lib):
            if not func_name.startswith('_'):
                print(f"  - {func_name}")
                
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)
