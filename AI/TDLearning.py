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
                'eval' : self.utility(newState),
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
        pass
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

        return 
        
    def getReward(self, hasWon):
        if hasWon:
            return 1
        if not hasWon:
            return -1
        else:
            return -0.01

class TDLearningTest(unittest.TestCase):

    #queens, anthills, and tunnels only
    def testUtilityBasic(self):
        player = AIPlayer(0)
        gameState = GameState.getBasicState()

        self.assertEqual(player.utility(gameState), 0.01)

    def testBestMove(self):
        player = AIPlayer(0)

        nodes = []

        #making node objects (only eval is used)
        for i in range(10):
            node = {
                'eval' : i,
            }
            nodes.append(node) 

        best = player.bestMove(nodes)

        self.assertEqual(best['eval'], 9)

    def testInitWeights(self):
        player = AIPlayer(0)
        hiddenLayer = player.initWeights(40)
        outputLayer = player.initWeights(9)

        for num in hiddenLayer:
            self.assertAlmostEqual(num, 0, delta=1)
        
        for num in outputLayer:
            self.assertAlmostEqual(num, 0, delta=1)
        
        self.assertEqual(40, len(hiddenLayer))
        self.assertEqual(9, len(outputLayer))

    def testSigmoid(self):
        player = AIPlayer(0)
        num = player.sig(1)
        self.assertAlmostEqual(num, 0.7310585786300049)

    def testActivateNeuron(self):
        player = AIPlayer(0)
        weights = [-0.5061572036196194, 0.11710044128933261, -0.5640861215824573, -0.6753749016864017, 0.20443410702909404]
        inputs = [1, 0, 0, 1]
        activation = player.activateNeuron(inputs, weights)
        self.assertAlmostEqual(activation, (-0.184622655301))
    
    def testGetOutputOneNeuron(self):
        player = AIPlayer(0)

        hiddenWeights = [-0.5061572036196194, 0.11710044128933261, -0.5640861215824573, -0.6753749016864017, 0.20443410702909404]
        outputWeights = [0.2395010298047295, -0.79178479274999917]
        inputs = [1, 0, 0, 1]

        output = player.getOutput(inputs, hiddenWeights, outputWeights)
        #CHECK to make sure this is the expected number
        self.assertAlmostEqual(output, 0.4700, delta=0.0001)

    def testGetOutputEightNeurons(self):
        player = AIPlayer(0)

        #40 weights for 8 neurons
        hiddenWeights = [0.3415, -0.4910, 0.7999, 0.1322, -0.9931, 
                         0.5132, -0.1122, -0.8483, 0.6340, 0.8888,
                         0.1342, -0.9348, -0.1234, 0.4333, -0.1222,
                         0.3937, -0.3882, 0.5555, 0.9294, 0.8726,
                         0.3947, 0.9673, 0.4872, -0.8366, -0.2838,
                         0.6333, -0.4522, 0.9983, 0.8272, 0.2333,
                         0.3344, -0.5523, -0.9101, 0.3710, 0.3999,
                         -0.1233, -0.3456, -0.3291, -0.9967, -0.8437]

        #9 weights for one output
        outputWeights = [0.2334, -0.2985, 0.9090, 0.7329, 0.1121,
                         0.1022, -0.5234, -0.6444, -0.7291]

        inputs = [1, 0, 0, 1]

        #round to 4 places, it's close enough after testing with 3 different sets of numbers
        aiOutput = round(player.getOutput(inputs, hiddenWeights, outputWeights), 4)

        #CHECK to make sure this is the expected number
        self.assertAlmostEqual(aiOutput, 0.6027, delta=0.001)

    def testGetErrorTerm(self):
        player = AIPlayer(0)
        #self.assertAlmostEqual(player.getErrorTerm(0, 0.4101), -0.0992, delta=0.0001)
        #self.assertAlmostEqual(player.getErrorTerm(1, 0.3820), -0.1457, delta=0.0001)
        self.assertAlmostEqual(player.getErrorTerm(-0.2650, 0.2650), -0.0516, delta=0.0001)

    def testGetHiddenNodeError(self):
        player = AIPlayer(0)

        errTerm = player.getErrorTerm(-0.2650, 0.2650)

        outputWeights = [0.2334, -0.2985, 0.9090, 0.7329, 0.1121,
                         0.1022, -0.5234, -0.6444, -0.7291]

        hiddenErrorList = player.getHiddenNodeError(errTerm, outputWeights)

        expectedList = [0.0154026, -0.0469044, -0.03781764, -0.00578436,
            -0.00527352, 0.02700744, 0.03325104, 0.03762156]

        for i in range(len(hiddenErrorList)):
            self.assertAlmostEqual(hiddenErrorList[i], expectedList[i], delta=0.0001)
    
    def testGetHiddenOutputList(self):
        player = AIPlayer(0)
        inp = [1, 0, 0, 1]
        hiddenWeights = [0.5061, 0.1171, -0.5640, -0.6753, 0.2044, 
                        0.1342, -0.4829, 0.8382, -0.3222, 0.0421]
        
        activationList = [0.8276, -0.3066]
        sigList = []
        for num in activationList:
            sigList.append(player.sig(num))

        self.assertEqual(player.getHiddenOutputList(inp, hiddenWeights), sigList)


    def testGetHiddenNodeErrorTerms(self):
        player = AIPlayer(0)

        inp = [1, 0, 0, 1]
        hiddenWeights = [0.5061, 0.1171, -0.5640, -0.6753, 0.2044, 
                        0.1342, -0.4829, 0.8382, -0.3222, 0.0421]

        #[0.695847223430284, 0.42394485650174163]
        hiddenOutputList = player.getHiddenOutputList(inp, hiddenWeights)
        

        hiddenErrorList = [0.0154, -0.0469]

        hiddenNodeErrorTermsList = player.getHiddenNodeErrorTerms(hiddenOutputList, hiddenErrorList)
        expectedList  = [0.0032, -0.0114]
        for i in range(len(hiddenErrorList)):
            self.assertAlmostEqual(hiddenNodeErrorTermsList[i], expectedList[i], delta=0.0001)

    def testAdjustWeight(self):
        player = AIPlayer(0)
        self.assertAlmostEqual(player.adjustWeight(0.1, 0.0075, 1), 0.1038, delta=0.0001)

    def testGetNodeIndex(self):
        player = AIPlayer(0)
        #changing to account for 9 weights
        self.assertEqual(player.getNodeIndex(3), 0)
        self.assertEqual(player.getNodeIndex(5), 0)
        self.assertEqual(player.getNodeIndex(14), 1)
        self.assertEqual(player.getNodeIndex(17), 1)
        self.assertEqual(player.getNodeIndex(23), 2)

    def testBackPropagate(self):
        #40 weights for 8 neurons
        hiddenWeights = [0.3415, -0.4910, 0.7999, 0.1322, -0.9931, 
                         0.5132, -0.1122, -0.8483, 0.6340, 0.8888,
                         0.1342, -0.9348, -0.1234, 0.4333, -0.1222,
                         0.3937, -0.3882, 0.5555, 0.9294, 0.8726,
                         0.3947, 0.9673, 0.4872, -0.8366, -0.2838,
                         0.6333, -0.4522, 0.9983, 0.8272, 0.2333,
                         0.3344, -0.5523, -0.9101, 0.3710, 0.3999,
                         -0.1233, -0.3456, -0.3291, -0.9967, -0.8437]

        #9 weights for one output
        outputWeights = [0.2334, -0.2985, 0.9090, 0.7329, 0.1121,
                         0.1022, -0.5234, -0.6444, -0.7291]

