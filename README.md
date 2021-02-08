Fighting Game Reinforcement Learning
======================

# 1. 개요(Overview)

저는 텐서플로우2 프레임워크의 에이전트 라이브러리를 사용하여
파이썬 강화학습 환경을 만든 뒤 언리얼 엔진4로 작성된 슈팅게임을 TCP 소켓을 통해 전달 받은
인풋값으로 에이전트를 학습하는 프로젝트를 진행한 바가 있습니다.

I implemented the reinforcement learning agent that learn the simple shooting game.
In this project, I connect shooting game that is created by UE4 with Python RL Server
through using TCP Connector plugin in UE4.

[Shooting_Game_Learning_project](https://www.youtube.com/watch?v=PkrSou_82iI&ab_channel=HeMaKim, "link")

그러나 이 프로젝트에는 치명적인 단점이 존재하는데
바로 학습을 시키기 위해서 언리얼 엔진4로 작성된 게임을 장시간 실행시켜야 한다는 점 입니다.
보통 RL에서 어느정도 의미가 있는 결과를 보이기 위해서는 최소 에피소드를 10000회 이상 돌려봐야 합니다.
그러나 기존의 방식으로는 5천회도 버거운 수준이였습니다.
제가 가지고 있는 컴퓨터의 성능으로 5천회 가량의 에피소드를 소화하는데에만 약 5~6시간 이상이 소요되었고
그 이상 진행하려고하면 렉 현상이 발생하는 등 정상적으로 학습을 진행하기가 어려워졌기 때문입니다.

However, this shooting game learning project has a serious problem.
The problem is that the game must execude during long time for learning.
Usually, for showing signifiacnt result in Reinforcement learning, 
the agent should learn in at least 10000 episodes.
Unfortunatlely, I met limit in this project when about 5000 episodes beacause for learning about 5000 episodes, I spent a 5~6 hours.
I want to teach my agent for a long time, but my computer's perofrmance don't be follow my expectation.

따라서 저는 다음과 같은 방법을 생각해보았습니다.
> '가벼운 임의의 가상환경에서 학습을 시킨 뒤, 언리얼 엔진4로 학습 결과를 시각화해보면 어떨까?'

For this reason, I thought the method.
> 'Before play the game, the agent firstly learn the game in abstracted environment.'

![구상](/RLFighter/image.PNG)

Google Drive의 확장기능으로 사용할 수 있는 Colab은 구글 클라우드 환경에서 동작하는 파이썬 개발환경입니다.
하루 12시간 할당량 제한이 있지만 구글에서 제공하는 TPU, GPU자원을 무료로 사용할 수 있습니다.

따라서 저는 실제 게임의 내용을 어느정도 추상화한 가상환경을 Colab에서 학습시키고,
이렇게 학습된 강화학습 에이전트의 학습 결과를 언리얼 엔진 4를 보여주는 방식으로 프로젝트를 진행했습니다.

I implemented the abstracted game envrionment that is created by Python on the Colab and the agent will play the game that is implemented by UE4 after learning in the abstracted environment.

======================

# 2. 상세(Detail)

이번 프로젝트에서 에이전트가 학습할 게임은 RL_Fighter입니다.
인게임에서는 다음과같이 두명의 마네킹 캐릭터가 마주보고 선채로 서로 기술을 주고받는 격투게임입니다.
오른쪽 캐릭터가 에이전트이며 왼쪽의 캐릭터는 에이전트가 상대해야할 Enemy입니다.
Enemy는 모든 액션을 랜덤하게 수행합니다.
(후에 영상을 보시면 아시겠지만 영상 후반부에는 제가 직접 플레이하여 에이전트를 상대하는 부분도 추가했습니다.)

The project name is RL_Fighter. In this project, two mannequin that face each other will fight. The right manneqin is the agent that learned the game by RL and the left mannequin is Enemy that randomly do action.
(In the video's part of 100000 episode, I will play the game as a enemy with the agent.)

======================

## 2.1 Action

Fighter가 수행할수있는 액션을 다음과 같습니다.

In this part, I will describe about the Fighter's actions.

======================

### 2.1.1 Normal Attack
Normal punch와 Normal Kick은 일반 공격으로 딜레이가 없습니다.
하지만 적에게 그로기를 줄 수 없고 데미지가 낮습니다.
인게임에서는 다음과 같이 나타납니다

Normal punch and normal kick is normal attack, this actions don't exist a delay but can't give to enemy a groggy. Also the damage is lower than smash attack.

![NormalPunch1](/RLFighter/normal_punch1.gif)
![NormalPunch2](/RLFighter/normal_punch2.gif)
![NormalKick1](/RLFighter/normal_kick1.gif)
![NormalKick2](/RLFighter/normal_kick2.gif)

======================

### 2.1.2 Smash Attack
Smash punch와 Smash Kick은 스매시 공격으로 딜레이가 존재합니다.
사용 직후 Fighter는 일정 시간동안 아무런 행동도 하지 못하게되는 단점이 존재하지만, 그만큼 데미지가 강력하고 적중 시 상대방을 그로기 상태로 몰아넣을 수 있습니다. 그로기 상태로 몰리게 되면 일정 시간동안 액션을 수행할 수 없게됩니다.

Smash punch and smash kick is smash attack, this actions exist a delay. After doing smash attack, the Fighter can't do anything during a delay but damage of the smash attack is higher than normal attack and can give a groggy. If the Fighter take a groggy, the Fighter can't do any action during a groggy time.

![SmashPunch1](/RLFighter/smash_punch1.gif)
![SmashPunch2](/RLFighter/smash_punch2.gif)
![SmashKick1](/RLFighter/smash_kick1.gif)
![SmashKick2](/RLFighter/smash_kick2.gif)
![SmashKick3](/RLFighter/smash_kick3.gif)

======================

<pre>
<code>
def do_action(self, idx):
    if idx == 0:
      return self.move_left()
    elif idx == 1:
      return self.move_right()
    elif idx == 2:
      return self.normal_punch()
    elif idx == 3:
      return self.normal_punch_2()
    elif idx == 4:
      return self.normal_kick()
    elif idx == 5:
      return self.normal_kick_2()
    elif idx == 6 :
      return self.smash_punch()
    elif idx == 7 :
      return self.smash_punch_2()
    elif idx == 8 :
      return self.smash_kick()
    elif idx == 9 :
      return self.smash_kick_2()
    elif idx == 10 :
      return self.smash_kick_3()
    elif idx == 11 :
      return self.guard()
</code>
</pre>

<pre>
<code>
def begin_action(self, actionIdx):
      print("Agents Action : ", actionIdx)
      agent_action_idx, agent_action_distance, agent_action_dmg, agent_delay, agent_groggy = self.agent_fighter.do_action(actionIdx)
      enemy_action = random.randint(0,11)
      print("Enemys Random Action : ", enemy_action)
      enemy_action_idx, enemy_action_distance, enemy_action_dmg, enemy_delay, enemy_groggy = self.enemy_fighter.do_action(enemy_action)

      self.judge_result(agent_action_idx, agent_action_distance, agent_action_dmg, agent_delay, self.enemy_fighter, agent_groggy)
      self.judge_result(enemy_action_idx, enemy_action_distance, enemy_action_dmg, enemy_delay, self.agent_fighter, enemy_groggy)
      pass
</code>
</pre>


![TCPConnector](/RLFighter/bp/StartConnect.PNG)
![TCPConnector](/RLFighter/bp/StartConnect2.PNG)
![TCPConnector](/RLFighter/bp/StartConnect3.PNG)

파이썬 가상환경에서는 Fighter 클래스의 doAction함수의 인자값을 통해서 실행됩니다. 이 doAction() 함수는 RLFighterEnvironment 클래스의 begin_action()함수에서 사용되는데, RLFighterEnvironment 클래스가 가진 Fighter 클래스 타입의 agent를 통해서 실행됩니다. 
인게임 환경에서는 TCP Connector가 파이썬 서버로부터 전달받은 액션값을 사용하여
해당 값에 맞는 액션을 Figther 클래스의 Agent 레퍼런스 변수를 통해서 실행하게 됩니다.
(각 액션에 대한 자세한 코드는 python 디렉토리에있는 코드를 참조해주세요. 블루프린트 역시 blueprint 디렉토리에 존재하는 이미지 파일로 참조할 수 있습니다.)

In the abstracted environment, Fighter's action execute through doAction() function.
The doAction() function is used in begin_action() function in RLFighterEnvironment class.
RLFighterEnvironment has the agent value as type of the Fight class and RLFighterEnvironment class call doAction() Function through the agent value. 
In the game RL TCP Connector receive the action value from the Python RL server and Connector operate a action through using reference value of the Fighter class.
(If you want to see about the detail of actions, please reference in python directory and blueprint directory.)


<pre>
<code>
def judge_result(self, _action_idx, _action_distance, _action_dmg, _delay, _target, _groggy):
      if _action_idx == -1: 
        # _action_idx = -1 -> 액션 수행 불가
        pass
      elif _action_idx == 0 or _action_idx == 1: 
        # _action_idx = 0,1 -> 이동판정
        pass
      elif _action_idx > 1 and _action_idx < 11: 
        # _action_idx = 2~5 -> 공격판정이므로 거리 데미지 딜레이 고려
        if _target.current_action == 11: # when guard
          print(_target.name,"이 가드상태입니다 이 공격은 무효화 됩니다.")
          pass
        else:
          if _action_distance > self.get_distance():
            _target.take_damage(_action_dmg, _groggy)
            _target.take_delay(_delay)
          else:
            pass
      elif _action_idx == 11: # when guard
        pass
      pass
</code>
</pre>


judge_result() 함수는 에이전트와 Enemy가 결정한 액션 결과에 따라서 판정결과를 리턴합니다.
공격을 수행했을 경우 공격의 거리, 그리고 공격이 적중했다면 가드를 했는지 여부를 판별한 뒤 데미지를 적용합니다.

The judge_result() function return a result about the action that is decided by the agent and the enemy. This function consider about distance of the attack and check about the taking guard action.


<pre>
<code>
def is_game_end(self):
      print("Action Result State : ", self._state)
      if self.agent_fighter.health < 0:
        print("enemy win")
        return 0
      elif self.enemy_fighter.health < 0:
        print("agent win")
        return 1
      elif self.num_turn >= 100:
        return 3
      else:
        print("game continue red의 체력 : ",self.agent_fighter.health, "blue의 체력", self.enemy_fighter.health, "현재 턴수는 : ", self.num_turn, "최단 기록은 :", self.best_record)
        return 2
</code>
</pre>


is_game_end함수를 통해서 게임이 계속 진행중인지 아니면 누가 승자인지를 판별합니다.
후에 이 함수는 보상값을 결정하는데 사용됩니다.

RLFighterEnvironment class decide the winner in the episode. If the winner is not decided RLFighterEnvironment class continue the episode.

보상은 패배했을때 -1을
그리고 승리했을때는 1 + 최단기록 갱신 보너스 + 퍼펙트 플레이 보너스 + 게임 스피드 보너스
로 구성했습니다.
각 보너스 함수는 다음과 같습니다.

The agent receive a reward in the episode.
when the agent lose
reward = -1
when the agent win 
reward = 1 + renew_best_record() + perfect_play_bonus() + speed_game_bonus()

<pre>
<code>
def renew_best_record(self):
      if self.num_turn <= self.best_record and self.num_turn != 0:
        self.best_record = self.num_turn
        return 5.0
      else:
        return 0.0
</code>
</pre>


최단 기록함수는 현재 에이전트가 갱신한 최단기록보다 작거나 같으면 5.0점을 추가하는 함수입니다.

If the agent renew shortest record during the episode, this function will give a 5.0 reward.

<pre>
<code>
def perfect_play_bonus(self):
      if self.agent_fighter.health == 100:
        return 3.0
      else:
        return 0.0
</code>
</pre>


퍼펙트 플레이 보너스함수는 에이전트가 체력 100을 유지한 상태로 승리하면 3.0점을 추가해주는 함수입니다.

If the agent win with 100 heath value, this function will give a 3.0 reward.

<pre>
<code>
def speed_game_bonus(self):
      game_speed = self.num_turn
      if game_speed <= 50:
        return 1.0
      elif game_speed <= 40:
        return 2.0
      elif game_speed <= 30:
        return 3.0
      elif game_speed <= 20:
        return 4.0
      elif game_speed <= 10:
        return 5.0
      elif game_speed <= 5:
        return 10.0
      else:
        return 0.0
</code>
</pre>


게임 스피드 보너스는 매번 체크된 에이전트의 turn수를 체크하여 turn이 낮을수록(게임을 빨리끝낼수록)더욱 많은 보상을 주는 함수입니다.

The agent receive bonus reward by the number of turns.


step함수는 다음과같습니다.
<pre>
<code>
def _step(self, action):
        print("Action Value :", action)
        groggy_state = 0
        delay_state = 0
        if self.agent_fighter.groggy_count > 0:
          groggy_state = 1
        else:
          groggy_state = 0

        if self.agent_fighter.delay > 0:
          delay_state = 1
        else:
          delay_state = 0
        
        print("Before State is : ", self._state)
        self._state = np.array([self.agent_fighter.health, self.enemy_fighter.health, self.get_distance(), self.agent_fighter.current_action, self.enemy_fighter.current_action, groggy_state, delay_state], dtype=np.int32)
        print("After State is : ", self._state)
        #self._state = convert_tensor(np.array([self.agent_fighter.health, self.enemy_fighter.health, self.get_distance(), self.agent_fighter.current_action, self.enemy_fighter.current_action, self.agent_fighter.groggy_count, self.agent_fighter.delay], dtype=np.int64))
        # This function is custom function
        # Write down your RL algorithm
        
        # when episode is ended
        if self._episode_ended:
            return self.reset()
        
        # self.send_action(str(action))
        self.begin_action(action)
        
        global is_rl_episode_end
        global agent_wins
        global agent_loses
        global each_agent_wins 
        global each_agent_loses
        global each_game_draw
        global game_draw

        if self.is_game_end() == 2: # continue game
            print("Game Continue State Data : ", self._state)
            print("And Recent Winrate . . . : ", each_agent_wins, "Loses : ", each_agent_loses, "Draws : ", each_game_draw)
            self.num_turn = self.num_turn + 1
            self.breaktime()
            return ts.transition(
            self._state, reward= 0.0, discount=1.0)
        elif self.is_game_end() == 1: # agent win
            agent_wins = agent_wins + 1
            each_agent_wins = each_agent_wins + 1
            print("Agent Win Recent Wins : ", each_agent_wins, "Loses : ", each_agent_loses, "Draws : ", each_game_draw)
            best_record_bonus_reward = self.renew_best_record()
            perfect_bonums_reward = self.perfect_play_bonus()
            speed_bonus = self.speed_game_bonus()
            agent_reward = 1.0 + speed_bonus + best_record_bonus_reward + perfect_bonums_reward
            self.game_reset()
            is_rl_episode_end = True
            print("Give Reward : ", agent_reward)
            print("State Data : ", self._state)
            return ts.termination(self._state, agent_reward)
        elif self.is_game_end() == 0: # enemy win
            agent_loses = agent_loses + 1
            each_agent_loses = each_agent_loses + 1
            print("Enemy Win Recent Wins : ", each_agent_wins, "Loses : ", each_agent_loses)
            self.game_reset()
            agent_reward = -1.0
            is_rl_episode_end = True
            print("Give Reward : ", agent_reward)
            print("State Data : ", self._state)
            return ts.termination(self._state, agent_reward)
        elif self.is_game_end() == 3:
            print("State Data : ", self._state)
            print("Game Draw")
            each_game_draw = each_game_draw + 1
            game_draw = game_draw + 1
            print("Agent Win Recent Wins : ", each_agent_wins, "Loses : ", each_agent_loses)
            self.game_reset()
            is_rl_episode_end = True
            return ts.termination(self._state, 0.0)
</code>
</pre>



3. 결과

Python abstracted environment result

![Result](/RLFighter/bp/abs_env_result.PNG)

※ 승률을 출력하는 함수부분에서 무승부를 카운팅 하는 부분에 문제가 있어서 결과가 정상적으로 출력되지 않을수있습니다. 후에 수정하겠습니다. (학습에는 영향을 미치지 않습니다.)
※ Winrate calculation function has a problem in counting the draw. I will modify it. (This problem don't affect about the agent learning.)

Video Link
[Result_Video](https://www.youtube.com/watch?v=RQ2Xax3P2iY&ab_channel=HeMaKim, "link")










