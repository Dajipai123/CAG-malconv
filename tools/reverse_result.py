import pefile
from capstone import *
from elftools.elf.elffile import ELFFile

# 加载PE文件
pe = pefile.PE('your_file_path')

# 获取CPU架构和位数
if pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_I386']:
    arch = CS_ARCH_X86
    mode = CS_MODE_32
elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_AMD64']:
    arch = CS_ARCH_X86
    mode = CS_MODE_64
elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_ARM']:
    arch = CS_ARCH_ARM
    mode = CS_MODE_ARM
elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_THUMB']:
    arch = CS_ARCH_ARM
    mode = CS_MODE_THUMB
elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_MIPSFPU']:
    arch = CS_ARCH_MIPS
    mode = CS_MODE_MIPS32
elif pe.FILE_HEADER.Machine == pefile.MACHINE_TYPE['IMAGE_FILE_MACHINE_MIPSFPU16']:
    arch = CS_ARCH_MIPS
    mode = CS_MODE_MIPS64
# 添加其他架构的判断...
else:
    raise Exception('Unsupported architecture')

# 创建Capstone实例
md = Cs(arch, mode)