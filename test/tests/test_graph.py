
import pprint
stacks = []
with open("test_file.txt", "r") as file:
    for line in file.readlines():
        if "search stack" in line:
            broken = line.split(":")
            stack = eval(broken[2])
            stacks.append(stack)

counts = {}
for stack in stacks:
    count = len(stack)
    if count in counts:
        counts[count] += 1
    else:
        counts[count] = 1

pprint.pprint(counts)