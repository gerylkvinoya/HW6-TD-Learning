##
# TDLearning Agent
# CS 421
#
# Authors: Geryl Vinoya and William Lau
#
# Sources Used:
#   Dr. Nuxoll's slides
##
'''
Yeah if you could pull from my branch and check if all the algorithms work. especially the updateStateUtility and getMove

and if those look good then try testing with a different reward function or a different alpha/discount factor
'''
import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
from typing import Dict, List
import unittest
import numpy as np
from pathlib import Path
import ast
import json


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "TDLearning")
        self.discountFactor = 0.9
        self.learningRate = 0.1

        #list to store the consolidated states in a tuple format
        #(categorizedState, utility)
        #maybe make a function that gets a categorized state, adds it to the list if it doesn't exist, and returns utility
        #if it already exists, return the utility from the tuple.
        self.states = self.readStateFile()


    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        moves = listAllLegalMoves(currentState)
        
        #this ensures that the current state is in the dict
        #currentUtility might not be used
        currentUtility = self.getStateUtility(currentState)

        stateUtilityList = []

        for move in moves:
            nextState = getNextStateAdversarial(currentState, move)
            nextUtility = self.getStateUtility(nextState)
            stateUtilityList.append((nextState, nextUtility, move))

        #get best utility from list, if there are ties, pick a random one
        random.shuffle(stateUtilityList)
        maxValue = max(stateUtilityList, key=lambda x: x[1])

        moveToMake = maxValue[2]
        newState = maxValue[0]

        if random.uniform(0, 1) < 0.05:
            randomValue = random.choice(stateUtilityList)
            moveToMake = randomValue[2]
            newState = randomValue[0]

        self.updateStateUtility(currentState, newState, self.getReward(currentState))

        return moveToMake


    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #need to update the current state's utility
        if hasWon:
            print("won")
        else:
            print("lost")
        self.outputStates()
        #get reward based on if it has won or not
        
    #reward
    #
    #Description: the reward function for TD learning
    #
    #Parameters:
    #   currentState - current state of the game (unused for now)
    #   hasWon - INTEGER value to determine if game is over, or in progress
    #       1 if won
    #       -1 if lost
    #       0 if nothing
    #
    #return: the node with the best eval
    def getReward(self, currentState):
        winner = getWinner(currentState)
        if winner == 1:
            return 100

        if winner == 0:
            return -100

        return self.rewardUtility(currentState)       

    def rewardUtility(self, currentState):
        WEIGHT = 10 #weight value for moves

        #will modify this toRet value based off of gamestate
        toRet = 0

        #get my id and enemy id
        me = currentState.whoseTurn
        enemy = 1 - me

        #get the values of the anthill, tunnel, and foodcount
        myTunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        myAnthill = getConstrList(currentState, me, (ANTHILL,))[0]
        myFoodList = getConstrList(currentState, 2, (FOOD,))
        enemyTunnel = getConstrList(currentState, enemy, (TUNNEL,))[0]

        #get my soldiers and workers
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        myWorkerList = getAntList(currentState, me, (WORKER,))

        #get enemy worker and queen
        enemyWorkerList = getAntList(currentState, enemy, (WORKER,))
        enemyQueenList = getAntList(currentState, enemy, (QUEEN,))

        for worker in myWorkerList:

            #if a worker is carrying food, go to tunnel
            if worker.carrying:
                tunnelDist = stepsToReach(currentState, worker.coords, myTunnel.coords)
                #anthillDist = stepsToReach(currentState, worker.coords, myAnthill.coords)

                #if tunnelDist <= anthillDist:
                toRet = toRet + (1 / (tunnelDist + (4 * WEIGHT)))
                #else:
                    #toRet = toRet + (1 / (anthillDist + (4 * WEIGHT)))

                #add to the eval if a worker is carrying food
                toRet = toRet + (1 / WEIGHT)

            #if a worker isn't carrying food, get to the food
            else:
                foodDist = 1000
                for food in myFoodList:
                    # Updates the distance if its less than the current distance
                    dist = stepsToReach(currentState, worker.coords, food.coords)
                    if (dist < foodDist):
                        foodDist = dist
                toRet = toRet + (1 / (foodDist + (4 * WEIGHT)))
        
        #try to get only 1 worker
        if len(myWorkerList) == 1:
            toRet = toRet + (2 / WEIGHT)
        

        #try to get only one soldier
        if len(mySoldiers) == 1:
            toRet = toRet + (WEIGHT * 0.2)
            enemyWorkerLength = len(enemyWorkerList)
            enemyQueenLength = len(enemyQueenList)
            
            #we want the soldier to go twoards the enemy tunnel/workers
            if enemyWorkerList:
                distToEnemyWorker = stepsToReach(currentState, mySoldiers[0].coords, enemyWorkerList[0].coords)
                distToEnemyTunnel = stepsToReach(currentState, mySoldiers[0].coords, enemyTunnel.coords)
                toRet = toRet + (1 / (distToEnemyWorker + (WEIGHT * 0.2))) + (1 / (distToEnemyTunnel + (WEIGHT * 0.5)))
            
            #reward the agent for killing enemy workers
            #try to kill the queen if enemy workers dead
            else:
                toRet = toRet + (2 * WEIGHT)
                if enemyQueenLength > 0:
                    enemyQueenDist = stepsToReach(currentState, mySoldiers[0].coords, enemyQueenList[0].coords)
                    toRet = toRet + (1 / (1 + enemyQueenDist))
            

            toRet = toRet + (1 / (enemyWorkerLength + 1)) + (1 / (enemyQueenLength + 1))

        #try to get higher food score
        foodCount = currentState.inventories[me].foodCount
        toRet = toRet + foodCount

        return toRet 
        
    
    #categorizeState
    #
    #Description: catergorizes a state based off of certain information in the state
    #
    #Parameters: currentState - state to categorize
    #
    #return: the category of the state in a Dict object
    def categorizeState(self, currentState):
        me = currentState.whoseTurn
        enemy = 1 - me

        myInv = currentState.inventories[me]
        myQueen = myInv.getQueen()

        #get the values of the anthill, tunnel, and foodcount
        myTunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        myAnthill = getConstrList(currentState, me, (ANTHILL,))[0]
        myFoodList = getConstrList(currentState, 2, (FOOD,))
        enemyTunnel = getConstrList(currentState, enemy, (TUNNEL,))[0]

        #get my soldiers and workers
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        myWorkerList = getAntList(currentState, me, (WORKER,))

        #get enemy worker and queen
        enemyWorkerList = getAntList(currentState, enemy, (WORKER,))
        enemyQueenList = getAntList(currentState, enemy, (QUEEN,))

        category = (
            currentState.inventories[me].foodCount, #my current food count
            self.queenOnBldg(myQueen, myTunnel, myAnthill), #true if queen is on a my tunnel/anthill, false if not
            len(mySoldiers), #number of soldiers
            len(myWorkerList), #number of workers
            self.carryingWorkerDist(myWorkerList, myTunnel, myAnthill), #minimum distance that any carrying worker is from a building
            self.nonCarryingWorkerDist(myWorkerList, myFoodList) #minimum distance that any carrying worker is from a building
        )

        return category

    #queenOnBldg
    #
    #Description: return true if the queen is on a building, false if not
    #
    #Parameters:
    #   myQueen
    #   myTunnel
    #   myAnthill
    #
    #return: boolean if queen is on a building
    def queenOnBldg(self, myQueen, myTunnel, myAnthill):
        if (myQueen.coords == myTunnel.coords) or (myQueen.coords == myAnthill.coords):
            return True
        return False

    #carryingWorkerDist
    #
    #Description: returns the minimum distance any carrying worker is from a building
    #
    #Parameters:
    #   myWorkerList
    #   myTunnel
    #   myAnthill
    #
    #return: int
    def carryingWorkerDist(self, myWorkerList, myTunnel, myAnthill):
        carryingWorkers = []
        distList = []
        for worker in myWorkerList:
            if worker.carrying:
                carryingWorkers.append(worker)
                tunnelDist = approxDist(worker.coords, myTunnel.coords)
                anthillDist = approxDist(worker.coords, myAnthill.coords)
                distList.append(min(tunnelDist, anthillDist))
            
        if len(carryingWorkers) == 0:
            return -1

        return min(distList)

    #nonCarryingWorkerDist
    #
    #Description: returns the minimum distance any carrying worker is from food
    #
    #Parameters:
    #   myWorkerList
    #   myTunnel
    #   myAnthill
    #
    #return: int
    def nonCarryingWorkerDist(self, myWorkerList, myFoodList):
        nonCarryingWorkers = []
        foodDist = []
        for worker in myWorkerList:
            if not worker.carrying:
                nonCarryingWorkers.append(worker)
                for food in myFoodList:
                    foodDist.append(approxDist(worker.coords, food.coords))

            
        if len(nonCarryingWorkers) == 0:
            return -1

        return min(foodDist)

    #readStateFile
    #
    #Description: reads contents of the states file into a list
    #
    #Parameters:
    #   
    #
    #return: list
    def readStateFile(self):
        stateDict = {}

        path = Path("./vinoya21_lauw22_states.txt")

        if path.is_file():
            f = open(path, 'r')
            data = f.read()

            stateDict = ast.literal_eval(data)

            f.close()

        return stateDict

    #outputStates
    #
    #Description: outputs the consolidated states to the text file
    #
    #Parameters:
    #   
    #
    #return: utility (float)
    def outputStates(self):
        f = open("AI/vinoya21_lauw22_states.txt", "w")
        f.truncate(0)

        f.write(str(self.states))

        f.close()

    #updateStateUtility
    #
    #Description: updates the utility of a state
    #             using equation U(A) = U(A) + alpha[R(A) + discount*U(B) - U(A)]
    #
    #Parameters:
    #   parentState - the previous state we are coming from
    #   currentState - the current state to evaluate
    #
    #return: None
    def updateStateUtility(self, currentState, nextState, reward):
        category = self.categorizeState(currentState)
        currentUtil = self.getStateUtility(currentState)
        nextUtil = self.getStateUtility(nextState)

        newUtil = currentUtil + self.learningRate*(reward + (self.discountFactor*nextUtil) - currentUtil)

        #because we're calling getStateUtility on currentState, we know that the category exists in the dictionary
        self.states[category] = newUtil

        return

    #getStateUtility
    #
    #Description: gets the utility of a state
    #
    #Parameters:
    #   currentState - the current state to get the utility from in self.states
    #
    #return: the state's utility from the dict
    def getStateUtility(self, currentState):
        category = self.categorizeState(currentState)

        if category in self.states:
            return self.states[category]

        #if we don't find it, add to the dict
        utility = 0.0
        self.states[category] = utility

        return utility

    #+100 for win -100 for losing +1 food count goes up or -1 food count goes down . -10 when ant dies ... strategies +10 when enemy dies / Reward +1 reward for when worker gains a food / state when agent has 2 workers or more than 2 workers penalty favor states/ examples...
        #alpha 0.1
        # discount factor - 0.8
        # explore and exploit - simple 0.01 chance to take a action or 0.05 implementation
        # Q learning - works but takes longer time (state and action i know next state) utilty of s prime / utility learning
        #     Q(s,a) = Q(s,a) + alpha * [ R(s) + discount * U(s') - Q(s,a)]
        #  start with reward function / then do tdlearning funciton
        # huge table with a lot of new states
        # output a 1 or 0 if we win or lose
        # power law of learning

    #Questions to ask:
    #How is the reward function going to work? Is it only used at the end of the game?
    #I am trying to save a q table to a file but can't. I previously saved each categorized state and utility and was successful but was unable to do it for a state/action pair
    #I understand what the idea of Q learning is, but confused about the actual Q table and how/when it should be updated
    #Ask to walk through the Q learning algorithm like in the textbook

    #get all moves, get next state
    #initialize all nextstates
    #generate all possible next states (with an action)
    #What's the utility of each state, should be initialized at 0
    #find the action leading to a state with the best utility (if tie, choose random)
    #no more than 5% chance to pick a random action for exploration
    #
    #update value of currentState w/ TDLearning
    #Going from state A to B, taking action a
    #U(A) = U(A) + alpha[R(A) + discount* U(B) - U(A)]
    #take the action selected and go to step 1
    #reward function
    #maybe +100 for win -100 for lose +10 for food++, +5 every time ant carry, +5 enemy at count--
    #use get nextstateadversarial

    #for each nextState, add to list of states
    #for each state, calculate utilities and add to list of utilities

    
        

    
#python -m unittest TDLearning.TDLearningTest
class TDLearningTest(unittest.TestCase):

    def testCategorizeState(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        category = player.categorizeState(gameState)
        #util = player.utility(gameState)

        #player.consolidatedStates.append((category, util))
        
    def testQueenOnBldg(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        me = gameState.whoseTurn
        enemy = 1 - me

        myInv = gameState.inventories[me]
        myQueen = myInv.getQueen()

        #get the values of the anthill, tunnel, and foodcount
        myTunnel = getConstrList(gameState, me, (TUNNEL,))[0]
        myAnthill = getConstrList(gameState, me, (ANTHILL,))[0]

        self.assertEqual(player.queenOnBldg(myQueen, myTunnel, myAnthill), True)

    def testCarryingWorkerDist(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        me = gameState.whoseTurn
        enemy = 1 - me

        myTunnel = getConstrList(gameState, me, (TUNNEL,))[0]
        myAnthill = getConstrList(gameState, me, (ANTHILL,))[0]

        

        worker1 = Ant((0, 2), WORKER, me)

        myWorkerList = []
        myWorkerList.append(worker1)

        self.assertEqual(player.carryingWorkerDist(myWorkerList, myTunnel, myAnthill), -1)

        worker2 = Ant((3, 0), WORKER,  me)
        worker2.carrying = True

        myWorkerList.append(worker2)

        self.assertEqual(player.carryingWorkerDist(myWorkerList, myTunnel, myAnthill), 3)

        worker3 = Ant((9, 1), WORKER, me)
        worker3.carrying = True

        myWorkerList.append(worker3)

        self.assertEqual(player.carryingWorkerDist(myWorkerList, myTunnel, myAnthill), 1)

    def testNonCarryingWorkerDist(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        me = gameState.whoseTurn

        food1 = Construction((0, 7), FOOD)
        food2 = Construction((3, 2), FOOD)
        foodList= []
        foodList.append(food1)
        foodList.append(food2)

        worker1 = Ant((0, 2), WORKER, me)

        myWorkerList = []

        self.assertEqual(player.nonCarryingWorkerDist(myWorkerList, foodList), -1)

        myWorkerList.append(worker1)

        self.assertEqual(player.nonCarryingWorkerDist(myWorkerList, foodList), 3)

        worker2 = Ant((1, 7), WORKER, me)
        myWorkerList.append(worker2)

        self.assertEqual(player.nonCarryingWorkerDist(myWorkerList, foodList), 1)

    def testGetReward(self):
        player = AIPlayer(0)
        currentState = GameState.getBasicState()

        self.assertEqual(player.getReward(currentState), -0.01)

    def testGetStateUtility(self):
        player = AIPlayer(0)
        currentState = GameState.getBasicState()

        self.assertEqual(player.getStateUtility(currentState), 0.0)

    def testUpdateStateUtility(self):
        player = AIPlayer(0)
        currentState = GameState.getBasicState()

        moves = listAllLegalMoves(currentState)
        #print(player.categorizeState(currentState))
        #use the first move in the list
        newState = getNextStateAdversarial(currentState, moves[1])
        #print(player.categorizeState(newState))

        player.updateStateUtility(currentState, newState, -100)

        

        #print(player.states)
        #print(player.states[player.categorizeState(newState)])