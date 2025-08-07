<<<<<<< HEAD
# ICRL
=======
baseline: 直接使用basemodel
ICRL: 对于同一个问题，每轮有估计出ground trouth 比如用投票，然后把每轮把这个估计出的结果当作ground trouth。在对于当前的整段回复给予奖励 正确是1 错误是0. 然后把这个 这个写进prompt里面 现在的prompt 应该是这样的格式{ 问题，round1 answer -reward}。然后把这些一起放进round2 在开启下一轮。
ICRL_wo_reward: 对于一个问题，只是把前一轮的回复放进prompt 应该是这样的格式{ 问题，round1 answer} 然后进行下一轮
ICRL: 现在有问题Q，输入：{Q} 进行k次完整推理(A11，A12，...A1k)，然后抓取出数字答案(a11,a12,...a1k), 对数字答案(a11,a12,...a1k)进行投票选出票数最多的答案a1. 然后在1-k随机选择一个回答（A1i，a1i）如果a1i == a1，A1i 对应reward=1，如果a1i ！= a1，A1i 对应reward=0。 然后修改提示词输入：{Q，A1i A1i_reward},进行k次完整推理(A21，A22，...A2k)，然后抓取出数字答案(a21,a22,...a2k)进行投票选出票数最多的答案a1.然后在1-k随机选择一个回答（A2i，a2i）如果a2i == a2，A2i 对应reward=1，如果a2i ！= a2，A2i 对应reward=0。然后修改提示词输入：{Q，A1i A1i_reward， A2i A2i_reward}
以此类推进行n轮，得到{Q，A1i A1i_reward， A2i A2i_reward，，，， Ani Ani_reward}
然后使用这个{Q，A1i A1i_reward， A2i A2i_reward，，，， Ani Ani_reward}来进行评估。

ICRL: 现在有问题Q，输入：{Q} 进行k次完整推理(A11，A12，...A1k)，然后抓取出数字答案(a11,a12,...a1k), 对数字答案(a11,a12,...a1k)进行投票选出票数最多的答案a1. 然后在1-k随机选择一个回答（A1i，a1i）如果a1i == a1，A1i 对应reward=1，如果a1i ！= a1，A1i 对应reward=0。 然后修改提示词输入：{Q，A1i A1i_reward},进行k次完整推理(A21，A22，...A2k)，然后抓取出数字答案(a21,a22,...a2k)进行投票选出票数最多的答案a1.然后在1-k随机选择一个回答（A2i，a2i）如果a2i == a2，A2i 对应reward=1，如果a2i ！= a2，A2i 对应reward=0。然后修改提示词输入：{Q，A1i A1i_reward， A2i A2i_reward}
以此类推进行n轮，得到{Q，A1i A1i_reward， A2i A2i_reward，，，， Ani Ani_reward}
然后使用这个{Q，A1i A1i_reward， A2i A2i_reward，，，， Ani Ani_reward}来进行评估。

ICRL0: 现在有问题Q，输入：{Q} 进行k次完整推理(A11，A12，...A1k)，然后抓取出数字答案(a11,a12,...a1k), 对数字答案(a11,a12,...a1k)进行投票选出票数最多的答案a1. 然后在1-k随机选择一个回答（A1i，a1i）如果a1i == a1，A1i 对应reward=1，如果a1i ！= a1，A1i 对应reward=0，然后按照一个如下提示词{生成解答的核心思路，可以忽略具体的数值计算，抓取推理逻辑，直接生成压缩后的回答，下面是需要压缩的回答：A1i }， 生成压缩后的回答为S1i，S1i的奖励保持不动和A1i相同假设是0 。 然后修改提示词输入：{Q，不好的回答reward=0{S1i}, 好的回答reward=1{}},进行k次完整推理(A21，A22，...A2k)，然后抓取出数字答案(a21,a22,...a2k)进行投票选出票数最多的答案a1.然后在1-k随机选择一个回答（A2i，a2i）如果a2i == a2，A2i 对应reward=1，如果a2i ！= a2，A2i 对应reward=0。然后按照一个如下提示词{生成解答的核心思路，可以忽略具体的数值计算，抓取推理逻辑，直接生成压缩后的回答，下面是需要压缩的回答：A2i }， 生成压缩后的回答为S2i，S2i的奖励保持不动和A2i相同假设是1 。然后修改提示词输入：{Q，不好的回答reward=0{S1i}, 好的回答reward=1{S2i}}
以此类推进行n轮，得到例如{Q，不好的回答reward=0{S1i,S3i，S4i...}, 好的回答reward=1{S2i,S5i....}}
然后使用这个{Q，不好的回答reward=0{S1i,S3i，S4i...}, 好的回答reward=1{S2i,S5i....}}来进行评估。




，进行投票得到ground trouth。在这k次回复中随机选择一个完整回复，如果该抽取的数字

对于同一个问题，每轮有估计出ground trouth 比如用投票，然后把每轮把这个估计出的结果当作ground trouth。在对于当前的整段回复给予奖励 正确是1 错误是0. 然后把现在回复写进prompt里面 现在的prompt 应该是这样的格式{问题, 好的回答(reward=1){} 坏的回答(reward=0){}},。然后把这些一起放进round2 在开启下一轮。






运行脚本
conda activate ICRL
bash /home/jovyan/ICRL/scripts/run_icrl.sh Qwen2.5-Math-7B AIME-TTT






1. 早停，简单的问题 （Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report）。 vote 集中就停 https://arxiv.org/abs/2406.07394
2. 看看 https://proceedings.mlr.press/v202/zhang23n/zhang23n.pdf 这篇文章。follow up看看 APPO，算法2
。用reward 距离进行打分https://arxiv.org/pdf/2402.09401
3. 时间消耗要和线形轮数增加，查看一下结果
4. 具体实现，错误1. 生成的回答和历史 逐个比较，关心的是data本身的diversity
这个距离怎么定义的？
A1， A2，A3

B1，B2，B3, B0 用B0看看早停

A1，A2，A3，B1 5 [10 10 12 13 10}
A1，A2，A3，B2 5 [10 10 10 5 10}

或者最小值或者mean或者最大值
||A1-B1|| +||A2-B1|| +||A3-B1|| /3 
||A1-B2|| +||A2-B2||+ ||A3-B2|| /3









7 {} 5000
Answer1 
相似性计算。

数学上 找到一个 
现在 16times random samlpe 1
>>>>>>> 4fe59c7 (Initial commit)
