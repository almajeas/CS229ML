import numpy as np

OFF_POLICY = 0
SARSA = 1
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
lr = 1
y = 0.9
PENALTY  = -1

class Node:
    def __init__(self,reward=0, up=0.0, right=0.0, down=0.0, left=0.0):
        self.values = np.array([up, right, down, left], dtype=np.dtype('Float64'))
        self.reward = reward
    def get_max_value(self):
        return np.amax(self.values)
    def get_max_direction(self):
        maximum_locations = np.where(self.values == np.amax(self.values))
        return maximum_locations[0]
    def __str__(self):
        s = f"{self.values}"
        return s

class RL:
    def __init__(self, mode=OFF_POLICY, epochs=1000, epsilon=0.2):
        self.mode = mode
        self.debug = False
        self.epsilon = epsilon
        self.epochs = epochs
        self.init_message()
        self.rows, self.columns = 4, 5
        self.grid = np.full((self.rows,self.columns),Node(), dtype=Node)
        rows, columns = self.grid.shape
        for row in range(rows):
            for col in range(columns):
                self.grid[row, col] = Node()
        self.grid[0,0] = Node(reward=10)
        self.grid[1,3] = Node(reward=-5)
        self.start = (self.rows-1, self.columns-1)
        self.current_loc = self.start
    def init_message(self):
        if self.mode == SARSA:
            print(f"Initializing... mode = SARSA, epochs = {self.epochs}, debug = {self.debug}")
        else:
            print(f"Initializing... mode = QLearning OffPolicy, epochs = {self.epochs}, epsilon = {self.epsilon} debug = {self.debug}")
    def p(self):
        rows, columns = self.grid.shape
        for row in range(rows):
            s = ""
            for col in range(columns):
                s += f"{self.grid[row, col]} "
            print(s)
    def step(self):
        # if self.mode == SARSA and np.random.rand() < self.epsilon:
        #     selected_direction = np.random.randint(0,len(self.grid[self.current_loc].values))
        # else:
        #     directions = self.grid[self.current_loc].get_max_direction()
        #     selected_direction = directions[np.random.randint(0,len(directions))]
        # side, newLoc , reward = self.get_next_loc(selected_direction)
        # if self.debug:
        #     print(f"{self.grid[self.current_loc].values[selected_direction]} + {lr} * ({reward} + {y}*{self.grid[newLoc].get_max_value()}")
        #     print(f"{self.grid[self.current_loc].values[selected_direction]}")
        # if self.train:
        #     self.grid[self.current_loc].values[selected_direction] += lr * (reward +  y*self.grid[newLoc].get_max_value()  - self.grid[self.current_loc].values[selected_direction])
        if self.mode == OFF_POLICY :
            if np.random.rand() < self.epsilon:
                selected_direction = np.random.randint(0,len(self.grid[self.current_loc].values))
            else:
                directions = self.grid[self.current_loc].get_max_direction()
                selected_direction = directions[np.random.randint(0,len(directions))]
            side, newLoc , reward = self.get_next_loc(selected_direction)
            if self.train:
                self.grid[self.current_loc].values[selected_direction] += lr * (reward +  y*self.grid[newLoc].get_max_value()  - self.grid[self.current_loc].values[selected_direction])
        else:
            if np.random.rand() < self.epsilon:
                selected_direction = np.random.randint(0,len(self.grid[self.current_loc].values))
            else:
                directions = self.grid[self.current_loc].get_max_direction()
                selected_direction = directions[np.random.randint(0,len(directions))]
            side, newLoc , reward = self.get_next_loc(selected_direction)
            if self.train:
                if np.random.rand() < self.epsilon:
                    self.grid[self.current_loc].values[selected_direction] += lr * (reward +  y*self.grid[newLoc].values[np.random.randint(0,4)]  - self.grid[self.current_loc].values[selected_direction])
                else:
                    self.grid[self.current_loc].values[selected_direction] += lr * (reward +  y*self.grid[newLoc].get_max_value()  - self.grid[self.current_loc].values[selected_direction])
        if self.debug:
            print(f"{self.grid[self.current_loc].values[selected_direction]} + {lr} * ({reward} + {y}*{self.grid[newLoc].get_max_value()}")
            print(f"{self.grid[self.current_loc].values[selected_direction]}")
        Qstar = self.grid[self.current_loc].values[selected_direction]
        loc = self.current_loc
        if newLoc == (0,0) :#or newLoc == (1,3):
            self.current_loc = self.start
            return loc, side, newLoc, Qstar
        self.current_loc = newLoc
        return loc, side, newLoc, Qstar
    def get_next_loc(self,direction):
        row, col = self.current_loc
        if direction == UP:
            if row == 0:
                return "UP", self.current_loc, PENALTY
            else:
                return "UP", (row-1, col), self.grid[(row-1, col)].reward
        if direction == RIGHT:
            if col == self.columns - 1:
                return "RIGHT", self.current_loc, PENALTY
            else:
                return "RIGHT", (row, col+1), self.grid[(row, col+1)].reward
        if direction == DOWN:
            if row == self.rows - 1:
                return "DOWN", self.current_loc, PENALTY
            else:
                return "DOWN", (row+1, col), self.grid[row+1, col].reward
        if direction == LEFT:
            if col == 0:
                return "LEFT", self.current_loc, PENALTY
            else:
                return "LEFT", (row, col-1), self.grid[(row, col-1)].reward
    def learn(self):
        self.train = True
        for _ in range(self.epochs):
            self.step()
        self.trainl = False
    def print_result(self):
        self.mode = OFF_POLICY
        self.epsilon = 0
        self.current_loc = self.start
        s, l = self.start
        while l != (0,0):
            loc, s, l, qstar = self.step()
            print(f"from {loc} go {s} to {l} - Q* = {qstar:0.2f}")
        self.p()
    def print_Q_summary(self):
        self.mode = OFF_POLICY
        self.epsilon = 0
        print("Q* Summary: ")
        for row in range(self.rows):
            for col in range(self.columns):
                s = f"Q*({(row, col)}) => "
                s += f"Up: {self.grid[row, col].values[UP]:0.2f}, "
                s += f"Right: {self.grid[row, col].values[RIGHT]:0.2f}, "
                s += f"Down: {self.grid[row, col].values[DOWN]:0.2f}, "
                s += f"Left: {self.grid[row, col].values[LEFT]:0.2f}"
                print(s)
    def print_V_summary(self):
        self.mode = OFF_POLICY
        self.epsilon = 0
        print("V* Summary: ")
        vals = np.empty((self.rows, self.columns))
        directions = np.empty((self.rows, self.columns),dtype=str)
        for row in range(self.rows):            
            for col in range(self.columns):
                direction = self.grid[row, col].get_max_direction()[0]
                value = self.grid[row, col].get_max_value()
                s = f"V*({(row, col)}) => {self.direction_string(direction)} = {value:0.2f}"
                vals[row][col] = f"{value:0.2f}"
                directions[row][col] =  self.direction_string(direction)
                # print(s)
        print(vals)
        print(directions)
    def direction_string(self, direction):
        if direction == UP:
            return "Up"
        elif direction == RIGHT:
            return "Right"
        elif direction == DOWN:
            return "Down"
        elif direction == LEFT:
            return "Left"
        else:
            return "Unknown Direction"
q = RL(mode=OFF_POLICY, epochs=20000)
q.learn()
q.print_result()
q.print_Q_summary()
q.print_V_summary()

q = RL(mode=SARSA, epochs=200000, epsilon=0.1)
q.learn()
q.print_result()
q.print_Q_summary()
q.print_V_summary()





