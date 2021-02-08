#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from socket import *
import threading
import time

def ConvertUE4DataToState(__dataStr):
    dataStr = str(__dataStr)
    dataStr2 = dataStr.replace("b","")
    dataStr3 = dataStr2.replace("'","")
    splitedStr = dataStr3.split("|")
    if len(splitedStr) > 1: # >= 1
        print("Case 0 Debug : ", splitedStr)
        isdoneStr = splitedStr[0]
        stateStr = splitedStr[1].split(",")
        isdone = int(float(isdoneStr))
        state = []
        for x in stateStr:
            state.append(int(x))
        stateArr = np.array(state, dtype=np.int32)
    else:
        print("Case 1 Debug : ", splitedStr)
        isdone = 0
        stateArr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    return isdone, stateArr

class RLAgentServer:
    
    def __init__(self, _isdone, _stateArr, _learning_iteration):
        
        self.learning_iteration = _learning_iteration
        self.learning_iteration_start = 0
        
        self.initIdx = 0
        self.isReceivedData = False
        self.port = 8081
        self.serverSock = socket(AF_INET, SOCK_STREAM)
        self.serverSock.bind(('', self.port))
        self.recvData = ""
        self.recvIsDone = _isdone
        # np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.float64) -> stateArr example
        self.recvStateArr = _stateArr
        
        self.isEpisodeEnd = False
        
    def send(self, sock, data):
            sock.send(data.encode('utf-8'))
            
            
    def read_data(self):
        return self.recvIsDone, self.recvStateArr
    

    def receive(self, sock):
        while True:
            print("receive Data . . . . .")
            recvData = sock.recv(1024)
            print("finish recv data")
            self.isReceivedData = True
            print("set data condition")
            self.recvData = ""
            print("set self data")
            #print('UE4 Data :', recvData.decode('utf-8'))
            self.recvData = recvData
            print("finish set self data")
            print(recvData)
            if self.initIdx > 0:
                __isdone, __state = ConvertUE4DataToState(self.recvData)
                print(type(__isdone), "isDone : ", __isdone)
                self.recvIsDone = __isdone
                print(type(__state), "State : ", __state)
                self.recvStateArr = __state
                #self.send(self.connectionSock, "1")
            self.initIdx = self.initIdx + 1
            print("RLServer : ", self.initIdx) 
        
    def run_server(self):
        self.serverSock.listen(1)
        print('%d port num wait ...'%self.port)
        self.connectionSock, self.addr = self.serverSock.accept()
        print(str(self.addr), 'accept UE4.')
        self.sender = threading.Thread(target=self.send, args=(self.connectionSock,))
        self.receiver = threading.Thread(target=self.receive, args=(self.connectionSock,))
        #self.sender.start()
        self.receiver.start()
        while True:
            time.sleep(1)
            pass


# In[ ]:




