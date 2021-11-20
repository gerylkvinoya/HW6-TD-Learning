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
        self.consolidatedStates = self.readStateFile()

        for state in self.consolidatedStates:
            print(state)
    
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
        self.gameStateList = []
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

        #self.move = Move
        #self.nextState = Gamestate
        #self.depth = 1
        #self.eval = Utility + self.depth
        #self.parent = None

        #create lists of all the moves and gameStates
        allMoves = listAllLegalMoves(currentState)
        self.gameStateList.append(currentState)
        stateList = []
        nodeList = []

        #for each move, get the resulting gamestate if we make that move and add it to the list
        for move in allMoves:

            if move.moveType == "END_TURN":
                continue

            newState = getNextState(currentState, move)

        
            stateList.append(newState)

            node = {
                'move' : move,
                'state' : newState,
                'depth' : 1,
                'eval' : self.processGamestate(newState),
                'parent': currentState
            }
                
            
            nodeList.append(node)
        
        #get the move with the best eval through the nodeList
        highestUtil = self.bestMove(nodeList)

        #return the move with the highest evaluation
        return highestUtil['move']

    
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
        self.outputStates()
        #get reward based on if it has won or not

    ##
    #utility
    #Description: examines GameState object and returns a heuristic guess of how
    #               "good" that game state is on a scale of 0 to 1
    #
    #               a player will win if his opponent’s queen is killed, his opponent's
    #               anthill is captured, or if the player collects 11 units of food
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: the "guess" of how good the game state is
    ##
    def utility(self, currentState):

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

                toRet = toRet + (1 / (tunnelDist + (4 * WEIGHT)))

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
                #distToEnemyTunnel = stepsToReach(currentState, mySoldiers[0].coords, enemyTunnel.coords)
                toRet = toRet + (1 / (distToEnemyWorker + (WEIGHT * 0.2)))# + (1 / (distToEnemyTunnel + (WEIGHT * 0.5)))
            
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

        #set the correct bounds for the toRet
        toRet = 1 - (1 / (toRet + 1))
        if toRet <= 0:
            toRet = 0.01
        if toRet >= 1:
            toRet = 0.99

        return toRet

    #bestMove
    #
    #Description: goes through each node in a list and finds the one with the 
    #highest evaluation
    #
    #Parameters: nodeList - the list of nodes you want to find the best eval for
    #
    #return: the node with the best eval
    def bestMove(self, nodeList):
        bestNode = nodeList[0]
        for node in nodeList:
            if (node['eval'] > bestNode['eval']):
                bestNode = node

        return bestNode

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

        category = {
                'foodCount' : currentState.inventories[me].foodCount, #my current food count
                'queenOnBldg' : self.queenOnBldg(myQueen, myTunnel, myAnthill), #true if queen is on a my tunnel/anthill, false if not
                'mySoldier' : len(mySoldiers), #number of workers
                'workerCount' : len(myWorkerList), #number of soldiers
                'carryingWorkerDist' : self.carryingWorkerDist(myWorkerList, myTunnel, myAnthill), #minimum distance that any carrying worker is from a building
                'nonCarryingWorkerDist': self.nonCarryingWorkerDist(myWorkerList, myFoodList) #minimum distance that any non carrying worker is from any of my food
            }

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

    #processGamestate
    #
    #Description: returns a utility of a consolidated state
    #
    #Parameters:
    #   currentState
    #
    #return: utility (float)
    def processGamestate(self, currentState):
        category = self.categorizeState(currentState)

        for state in self.consolidatedStates:
            if category == state[0]:
                return state[1]
        
        #if we don't find it, add to the list
        utility = self.utility(currentState)

        self.consolidatedStates.append((category, utility))

        return utility

    #readStateFile
    #
    #Description: reads contents of the states file into a list
    #
    #Parameters:
    #   
    #
    #return: list
    def readStateFile(self):
        stateUtilList = []

        path = Path("./vinoya21_lauw22_states.txt")

        if path.is_file():
            f = open(path, 'r')
            contents = f.read().splitlines()
            for line in contents:
                stateUtil = ast.literal_eval(line)
                stateUtilList.append(stateUtil)
            f.close()

        return stateUtilList

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
        for tuple in self.consolidatedStates:
            f.write(str(tuple) + "\n")
        f.close()

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
        

    
#python -m unittest TDLearning.TDLearningTest
class TDLearningTest(unittest.TestCase):

    def testCategorizeState(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        category = player.categorizeState(gameState)
        util = player.utility(gameState)

        player.consolidatedStates.append((category, util))

        #print(player.consolidatedStates)
        
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