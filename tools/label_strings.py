import subprocess
import re

def label_strings(strings):

    net_api_list = ['InternetOpen', 'InternetConnect', 'InternetOpenUrl', 'InternetReadFile', 'InternetWriteFile',
                    'InternetCloseHandle', 'HttpOpenRequest', 'HttpSendRequest', 'HttpQueryInfo', 'HttpAddRequestHeaders',
                    'socket', 'connect', 'send', 'recv', 'closesocket', 'WSAStartup', 'WSACleanup', 'WSAGetLastError',
                    'bind', 'listen', 'accept', 'sendto', 'recvfrom', 
                    ]

    sys_api_list = ['CreateFile', 'ReadFile', 'WriteFile', 'CloseHandle', 'RegOpenKey', 'RegQueryValue', 'RegSetValue', 'RegCloseKey',
                    'CreateProcess', 'CreateRemoteThread', 'CreateThread', 'LoadLibrary', 'GetProcAddress', 'VirtualAlloc', 'VirtualProtect',
                    'CreateFileMapping', 'MapViewOfFile', 'UnmapViewOfFile', 'CreateService', 'StartService', 'ControlService', 'DeleteService',
                    'CreateMutex', 'OpenMutex', 'CreateEvent', 'OpenEvent', 'CreateSemaphore', 'OpenSemaphore','GetComputerName', 'GetUserName',
                    'GetSystemDirectory', 'GetWindowsDirectory', 'GetTempPath', 'GetTempFileName', 'GetSystemMetrics', 'GetSystemInfo', 'GetSystemTime',
                    'GetLocalTime', 'GetTickCount', 'GetVersion', 'GetVersionEx', 'GetDriveType', 'GetDiskFreeSpace', 'GetDiskFreeSpaceEx', 'GetVolumeInformation',
                    'GetAdaptersInfo', 'GetAdaptersAddresses', 'GetIfTable', 'GetIfEntry', 'GetIpAddrTable', 'GetIpNetTable', 'GetIpForwardTable', 'GetUdpTable',
                    'SetWindowsHookEx', 'GetAsyncKeyState', 'GetKeyState', 'GetKeyboardState', 'GetKeyboardLayout', 'GetKeyboardLayoutList', 'GetKeyboardType',
                    ]
                    
    # 贴上网络api标签
    net_api = []
    for string in strings:
        for api in net_api_list:
            if api in string:
                net_api.append(string)

    # 贴上系统api标签
    sys_api = []
    for string in strings:
        for api in sys_api_list:
            if api in string:
                sys_api.append(string)

    # 导入文件列表
    end_list = ['dll', 'exe', 'sys', 'drv', 'ocx', 'vxd', 'cpl', 'scr', 'msc']
    import_file_list = []
    for end in end_list:
        end_regexp = re.compile(r'\.' + end + '$')
        for string in strings:
            if end_regexp.search(string):
                import_file_list.append(string)
    
    return net_api,sys_api,import_file_list


if __name__ == '__main__':
    file_name = '/home/fhh/ember-master/MalConv2-main/dataset/test2/mal_6fd4849beabb6b6d40230e9f4d491d26'
    # 静默执行
    strings = subprocess.run(['strings', file_name], stdout=subprocess.PIPE).stdout.decode('utf-8')
    strings = strings.split('\n')

    # 去重
    strings = list(set(strings))


    print(len(strings))
    net_api,sys_api,import_filelist = label_strings(strings)
    print(f'net_api:{net_api}')
    print(f'sys_api:{sys_api}')    
    print(f'import_filelist:{import_filelist}')



