##
# TDLearning Agent
# CS 421
#
# Authors: Geryl Vinoya and William Lau
#
# Sources Used:
#   Dr. Nuxoll's slides
##
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
        #currentUtility won't be used
        currentUtility = self.getStateUtility(currentState)

        #create a list that has each nextState, nextUtility, and move in a tuple
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

        #this is our chance to explore rather than exploit
        if random.uniform(0, 1) < 0.05:
            randomValue = random.choice(stateUtilityList)
            moveToMake = randomValue[2]
            newState = randomValue[0]

        #update the current state utility using the equation
        self.updateStateUtility(currentState, newState, self.getReward(currentState))

        #return our selected move
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
    # This agent learns
    #
    def registerWin(self, hasWon):
        #just needs to output states to the file at the end of each game.
        self.outputStates()
        
    #reward
    #
    #Description: the reward function for TD learning
    #
    #Parameters:
    #   currentState - current state of the game
    #
    #return: reward value
    def getReward(self, currentState):

        #if we win or lose add/subtract 100 for the reward
        winner = getWinner(currentState)
        if winner == 1:
            return 100
        if winner == 0:
            return -100

        #else, use a utility function as our reward
        return self.rewardUtility(currentState)       

    #rewardUtility
    #
    #Description: the reward helper function for TD learning
    #
    #Parameters:
    #   currentState - current state of the game
    #
    #return: reward value based on status of a current state
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

        #store the category into a tuple
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
    #return: nothing
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
        currentUtil = self.getStateUtility(currentState) #U(A)
        nextUtil = self.getStateUtility(nextState) #U(B)

        #TD learning equation
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

        #categorize the state
        category = self.categorizeState(currentState)

        #check if state already exists and return its utility
        if category in self.states:
            return self.states[category]

        #if we don't find it, add to the dict and return 0
        utility = 0.0
        self.states[category] = utility

        return utility
        
#Commands to run unit test:
#cd AI
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