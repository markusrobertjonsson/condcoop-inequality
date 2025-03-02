import matplotlib.pyplot as plt
import csv
import numpy as np

UC = "Unconditional Cooperator"
CC = "Conditional Cooperator"
FR = "Free-Rider"
NA = "Others"


def least_squares_regression(points):
    """
    Compute the least-squares regression line equation y = mx + b
    for a given set of (x, y) coordinates.

    Args:
        points (list of tuples): A list of (x, y) coordinates.

    Returns:
        tuple: (k, m) where y = kx + m is the regression line.
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # xx = np.sum((x - x_mean) ** 2)
    # print(f"xx={xx}")
    if np.sum((x - x_mean) ** 2) == 0:
        return None, None

    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    # plt.scatter(x, y)
    # plt.plot([0, 1], [intercept, intercept + slope])
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.show()

    return slope, intercept


# =OM(OCH(20*BK2+BL2<10;BL2<10);"FreeRider"; OM(OCH(20*BK2+BL2>=10;BL2>=10);"Contributor"; OM(OCH(BK2>0;BL2<10;20*BK2+BL2>10); "Conditional";"Others")))
def points2category(points):
    slope, intercept = least_squares_regression(points)
    if None in (slope, intercept):
        category = NA
    elif intercept < 0.5 and intercept + slope < 0.5:
        category = FR
    elif intercept >= 0.5 and intercept + slope >= 0.5:
        category = UC
    elif slope > 0 and intercept < 0.5 and intercept + slope > 0.5:
        category = CC
    else:
        category = NA
    return slope, intercept, category


class Player():
    def __init__(self, player_id, country, endowment):
        self.player_id = player_id
        self.country = country
        self.endowment = endowment
        self.contributions = [None] * 20
        self.slope = None
        self.intercept = None
        self.category = None

    def categorize(self, points):
        slope, intercept, category = points2category(points)
        self.slope = slope
        self.intercept = intercept
        self.category = category
        # print(f"{self.player_id}: coeff={self.slope}, intercept={self.intercept}")
        # input()


class Group():
    def __init__(self):
        self.players = dict()

    def contribution_sum(self, round, rel=True):
        '''Compute the sum of relative contributions in the specified round.'''
        s = 0
        for player in self.players.values():
            if rel:
                s += player.contributions[round] / player.endowment
            else:
                s += player.contributions[round]
        return s

    def categorize_players(self):
        assert(len(self.players) == 4)        
        for player in self.players.values():
            points = []
            for round in range(1, 20):
                rel_contribution = player.contributions[round] / player.endowment
                grp_sum_prev = self.contribution_sum(round - 1, rel=False)
                others_sum_prev = grp_sum_prev - player.contributions[round - 1]
                if player.endowment == 10:
                    others_relsum_prev = others_sum_prev / 70
                else:
                    others_relsum_prev = others_sum_prev / 50
                point = (others_relsum_prev, rel_contribution)
                points.append(point)
            # for round in range(1, 20):
            #     rel_contribution = player.contributions[round] / player.endowment
            #     rel_contribution_prev = player.contributions[round - 1] / player.endowment
            #     others_sum_prev = self.contribution_sum(round - 1) - rel_contribution_prev
            #     others_avg_prev = others_sum_prev / 3
            #     point = (others_avg_prev, rel_contribution)
            #     points.append(point)
            player.categorize(points)


players = dict()
groups = dict()
with open("data.csv", mode="r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file, delimiter=";")
    
    for row in reader:
        country = row['country']
        player_id = country + '_' + row['session'] + '_' + row['player']
        round = int(row['round'])
        contribution = int(row['contribution'])

        if player_id not in players:
            endowment = int(row['endowment'])
            player = Player(player_id, country, endowment)
            players[player_id] = player
        else:
            player = players[player_id]
        player.contributions[round - 1] = contribution

        group_id = country + '_' + row['session'] + '_' + row['group']
        if group_id not in groups:
            group = Group()
            groups[group_id] = group
        else:
            group = groups[group_id]

        if player_id not in group.players:
            group.players[player_id] = player

# Sanity checks
assert(len(players) == 4 * len(groups))
for group in groups.values():
    assert(len(group.players) == 4)
    endowments = [p.endowment for p in group.players.values()]
    assert(endowments.count(10) == 2)
    assert(endowments.count(30) == 2)
    ids = [id(p) for p in group.players.values()]
    assert(len(set(ids)) == len(ids))

ids = [id(p) for p in players.values()]
assert(len(set(ids)) == len(ids))
for player in players.values():
    assert(None not in player.contributions)

# Categorize each player in each group
for group in groups.values():
    group.categorize_players()

# Statistics
n_both = {UC: 0, CC: 0, FR: 0, NA: 0}
n_swe = {UC: 0, CC: 0, FR: 0, NA: 0}
n_sa = {UC: 0, CC: 0, FR: 0, NA: 0}
n_players = 0
n_players_swe = 0
n_players_sa = 0
for player in players.values():
    n_players += 1
    n_both[player.category] += 1
    if player.country == "Sweden":
        n_players_swe += 1
        n_swe[player.category] += 1
    elif player.country == "SouthAfrica":
        n_players_sa += 1
        n_sa[player.category] += 1
    else:
        assert(False)
assert(n_players == len(players))
assert(n_players_swe + n_players_sa == n_players)

for cat in n_both:
    n_both[cat] /= n_players
    n_swe[cat] /= n_players_swe
    n_sa[cat] /= n_players_sa

print("=== Both ===")
print(n_both)
print("=== Sweden ===")
print(n_swe)
print("=== SouthAfrica ===")
print(n_sa)

# Plot LCP lines
poor_cnt = {UC: 0, CC: 0, FR: 0}
rich_cnt = {UC: 0, CC: 0, FR: 0}
fig_poor, axes_poor = plt.subplots(1, 3, figsize=(12, 4))  # First figure
fig_rich, axes_rich = plt.subplots(1, 3, figsize=(12, 4))  # First figure
axes_poor = {UC: axes_poor[0], CC: axes_poor[1], FR: axes_poor[2]}
axes_rich = {UC: axes_rich[0], CC: axes_rich[1], FR: axes_rich[2]}
for player in players.values():
    if player.category == NA:
        continue
    x = [0, 1]
    y = [player.intercept, player.intercept + player.slope]
    if player.endowment == 10:
        poor_cnt[player.category] += 1
        axes_poor[player.category].plot(x, y)
    else:
        rich_cnt[player.category] += 1
        axes_rich[player.category].plot(x, y)
for cat, ax in axes_poor.items():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{cat} (n={poor_cnt[cat]})")
for cat, ax in axes_rich.items():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(cat)
    ax.set_title(f"{cat} (n={rich_cnt[cat]})")
fig_poor.suptitle("Poor")
fig_rich.suptitle("Rich")
plt.show()
