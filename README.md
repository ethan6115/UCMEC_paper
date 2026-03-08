# UCMEC-mmWave-Fronthaul
為了重現論文圖中的uplink隨cluster size的變化趨勢，只修正原版程式碼的部分錯誤，並且把環境參數值和論文的版本對齊。
論文原始程式碼:https://github.com/qlt315/UCMEC-mmWave-Fronthaul.git
# 和論文原版程式碼不同:
## 1. 程式邏輯錯誤(會對訓練有大影響)
### a. 用戶移動 (Mobility)：
* 舊版： 計算移動方向時多使用了np.abs，導致計算移動增量時方向錯誤（只能往x, y軸的正方向移動）。

```python
self.locations_users[i, 0] = self.locations_users[i, 0] + user_speed[i, 0] * np.abs(destination_users[i, 0] - self.locations_users[i, 0]) / slope
self.locations_users[i, 1] = self.locations_users[i, 1] + user_speed[i, 0] * np.abs(destination_users[i, 1] - self.locations_users[i, 1]) / slope
```
* 新版： dx = destination - current，直接保留正負號，確保用戶能正確地朝目標方向移動。
```python
self.locations_users[i, 0] = self.locations_users[i, 0] + user_speed[i, 0] * (destination_users[i, 0] - self.locations_users[i, 0]) / slope
self.locations_users[i, 1] = self.locations_users[i, 1] + user_speed[i, 0] * (destination_users[i, 1] - self.locations_users[i, 1]) / slope
```
### b. Episode Length錯誤設定
* 舊版： 雖然一個episode的長度為200，但是每經過20個step就會把環境設為done，reset函數中又沒有對step_num參數進行重置，導致不同episode的step_num步數會一直累加，且20步以後reset所輸出的agent觀測值全都是隨機值。

```python
if self.step_num > 20:
            done = [1] * self.M_sim
        else:
            done = [0] * self.M_sim
#reset 函數未重置step_num
def reset(self):
    sub_agent_obs = []
    for i in range(self.agent_num):
        sub_obs = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.obs_dim,))
        sub_agent_obs.append(sub_obs)
    return sub_agent_obs
```
* 新版： 將設為done的間隔和一個episode的長度同樣設為200，並且在reset函數中正確將episode長度reset成0
```python
if self.step_num >= 200:    #>20改>=200
    done = [1] * self.M_sim
else:
    done = [0] * self.M_sim

def reset(self):
    self.step_num = 0 
    ...
```
### c. Observation中Task的時間錯位
* 舊版: 問題在於 Agent 在 obs 內看到的是本步剛處理完的 Task(t-1)，卻要對下一步一個全新的 Task(t) 做決策，obs 與 action 的作用對象永遠錯開一步。
```
step(t-1) 開頭：
   生成 Task(t-1)（step 內的局部變數）
   用 Task(t-1) 計算 delay(t-1)、reward(t-1)
    ↓
step(t-1) 結尾建構 obs：
   obs = [Task(t-1), ω(t-1), p(t-1), delay(t-1)]
          ^^^^^^
          本步剛處理完的task
    ↓
Agent 收到 obs，依據 Task(t-1) 決策出 action(t)
    ↓
step(t) 開頭：
   生成 Task(t)（全新亂數，與 Task(t-1) 無關）
   action(t) 實際作用在 Task(t) 上
    ❌ Agent 決策時看的是 Task(t-1)，卻套用到完全不同的 Task(t)
```
* 新板: 將 Task 從局部變數改為成員變數，在每步結尾提前生成下一步的 Task(t+1) 並存入 self.Task_size，再放進 obs 回傳。如此一來，Agent 看到什麼 Task，action 就作用在那個 Task 上，符合標準 MDP 的狀態定義。
```
step(t-1) 開頭：
   用 self.Task_size（ = Task(t-1)）計算 delay(t-1)、reward(t-1)
    ↓
step(t-1) 結尾建構 obs：
   生成 Task(t)，更新 self.Task_size = Task(t)
   obs = [Task(t), ω(t-1), p(t-1), delay(t-1)]
          ^^^^^^
          下一步將要處理的 task
    ↓
Agent 收到 obs，依據 Task(t) 決策出 action(t)
    ↓
step(t) 開頭：
   用 self.Task_size（ = Task(t)）計算 delay(t)、reward(t)
   ✅ Agent 決策時看的是 Task(t)，action(t) 也作用在 Task(t) 上
```

## 2. 程式計算公式錯誤(bug)
### a. shadowing random方法錯誤(step()函式)
* shadowing的kappa是Gaussian distribution，因此把random從rand改成randn。
### b. link_type 索引錯誤(front_rate_cal()函式)
* 索引錯誤：[j, j] 應為 [i, j]，導致所有 user 的 LOS/NLOS 判斷都用同一個 AP 自己對自己的值，fronthaul rate 計算錯誤
### c. alpha_nlos 符號錯誤(front_delay_rate()函式)
* 論文版SINR_front_mole的alpha_nlos用正號，應為負號，導致 NLOS 路徑的衰減方向相反
### d. 中 SINR noise 累加錯誤(uplink_delay_rate()函式)
* 論文版SINR_access_noise在 cluster AP 迴圈內用 = 覆寫，應為 += 累加，導致只保留最後一個 AP 的 noise，SINR 分母被低估

## 3. 環境參數不同
論文和程式碼在某些環境參數上有部分差異，具體如下:

| 參數名稱 | 論文原版程式碼預設值 | 此程式碼版本 | 差異與影響分析 | 
| :--- | :--- | :--- | :--- | 
| **任務大小**<br>`Task_size` | **50 ~ 100 kbits**<br>(`50000` ~ `100000`) | **400 ~ 800 kbits**<br>(`50KB` ~ `100KB`) | **差距 8 倍**。<br>論文的任務大很多，傳輸延遲會暴增。 | 
| **本地算力**<br>`C_user` | **0.2 ~ 0.5 GHz**<br>(`2e8` ~ `5e8`) | **2 ~ 5 GHz**<br>(`2e9` ~ `5e9`) | **差距 10 倍**。<br>程式碼故意把本地算力設得很爛，**強迫 Agent 必須卸載**否則延遲會很高。<br>論文設定下，本地算力過強，Agent 容易全選本地運算。 |
| **訊號衰減閾值**<br>`d_1` | **50 m** | **15 m** | **關鍵差異**。<br>論文設定 (15m) 非常嚴苛，稍遠一點訊號就斷崖式下跌。<br>程式碼 (50m) 讓訊號覆蓋範圍大很多，連線品質較好。 | 
| **回合長度**<br>`Episode Length` | **20 步**<br>(`if step > 20: done`) | **200 步** | **嚴重 Bug**。<br>程式碼強制 20 步結束，導致 Agent 學不到長期策略。<br> |
| **存取頻寬**<br>`bandwidth_a` | **2 MHz** (`2e6`) | **20 MHz** | **差距 10 倍**。<br>舊版程式碼配合小任務用 2MHz。論文大任務配 20MHz。<br>兩者比例其實差不多。 | 
| **移動速度**<br>`user_speed` | **5-15 m/s**  | **10-20 m/s**<br> | 論文移動性較高，但對於整體影響不大。 |
| **AP cluster間隔**<br> | **每time step**  | **10 time step 一次**<br> | 實際實驗後，發現對環境的影響很小 |
| **天線陣列增益**<br> | **[Gm·Gm, Gm·Gs, Gs·Gs]**  | **[Gs·Gs, Gm·Gm, Gm·Gs]**<br> | 天線增益和各自對應的機率和論文中不同，目前統一改為論文的對應 |
| **total delay clip**<br> | **無clip**  | **clip到1秒**<br> | 之後根據實驗結果決定是否保留 |

## 4. 程式運算效率優化
* distance matrix：雙層巢狀迴圈改為 numpy broadcasting，一次計算所有 user-AP 距離
* pathloss：雙層巢狀迴圈改為 boolean mask 向量化，三個距離區間同時處理
* shadow fading & beta：雙層巢狀迴圈改為 matrix 直接運算，利用 broadcasting 自動展開 M×N 矩陣
* MMSE theta：雙層巢狀迴圈改為 matrix 直接運算，整個 M×N 一行計算完成
* cluster()：原本對每個 user 在 N_sim 次迴圈內重複執行相同的 argsort，改為每個 user 只算一次後直接取前 cluster_size 個
