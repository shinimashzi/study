# 【学习笔记】Threading

[TOC]

## 1. Add-添加线程

- 一些函数：

`threading.active_count()`： 运行的线程数

`threading.enumerate()`：列出当前活跃的线程

`threading.current_thread()`：显示当前线程

- 添加线程：

```python
def thread_job():
    print('thread 2, ', threading.current_thread())

def main():
    thread_added = threading.Thread(target=thread_job)
    thread_added.start()
```

## 2. Join

例子：

```python
import threading
import time
def thread_job():
    print('start \n')
    for i in range(10):
        time.sleep(0.1)
   # print('thread 2, ', threading.current_thread())
    print('T1 finish\n')
    
    
def main():
    thread_added = threading.Thread(target=thread_job, name='T1')
    thread_added.start()
    thread_added.join()
    print('all done\n')


if __name__ == '__main__':  
    main()
    
# start
# T1 finish
# all done

# if not execute join
# start
# all done
# T1 finish
```

`thread.join()`：等待`thread`运行结束，才会继续运行。

## 3. Queue

线程无法返回一个值，所以需要将运行中的数据放入`queue`中。

```python
import threading
import time
from queue import Queue


# q为将数据放入的队列
def job(l, q):
    for i in range(len(l)):
        l[i] = l[i]**2
    q.put(l)


def multi_threading():
    q = Queue()
    threads = [] 
    data = [[1,2,3], [3,4,5], [4,4,4], [5,5,5]]
    for i in range(4):
        t = threading.Thread(target=job, args=(data[i], q))
        t.start()
        threads.append(t)
    
    for thread in threads:
        thread.join()

    result = []
    for _ in range(4):
        result.append(q.get())
    print(result)

```

## 4. GIL-不一定效率更高

`python`**设计为在任意时刻只有一个线程在解释器中运行。** 

`GIL(Global Interpreter Lock)`:全局解释器锁，当执行多线程程序时，由`GIL`来控制同一时刻只有一个线程能够运行。

所以`python`中的多线程为表面多线程。

实现多线程的原理：**解释器分时复用** -> 并发

## 5. Lock

```python
import threading
from queue import Queue
import copy
import time


def job1():
    global A
    for i in range(10):
        A += 1
        print('job1', A)
        # time.sleep()

        
def job2():
    global A
    for i in range(10):
        A += 100
        print('job2', A)


if __name__ == '__main__':
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
```

使用`t1.start(),t2.start(),t1.join(),t2.join()`不能保证`t1`在`t2`前执行完，如果需要这样，则：

```python
import threading
from queue import Queue
import copy
import time


def job1():
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 1
        print('job1', A)
        time.sleep(0.1)
    lock.release()


def job2():
    global A, lock
    lock.acquire()
    for i in range(10):
        A += 100
        print('job2', A)
    lock.release()


if __name__ == '__main__':
    lock = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
```



