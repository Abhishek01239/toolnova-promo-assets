"""Narration for 'I Had 10 Minutes to Survive My First Night' (long-form, ~5 min)."""

TITLE = "I Had 10 Minutes to Survive My First Night"

CHAPTERS = [
    (1, "10 MINUTES OF DAYLIGHT", [
        "[excited] I spawned in a brand new world... and I had exactly TEN minutes before my first night.",
        "[determined] No house. No tools. No weapons. Just me, a forest, and a sun that was already moving.",
        "[curious] In Minecraft, the first night is where most players lose everything. Zombies... skeletons... and creepers.",
        "[excited] So I made a plan. Wood first. Then stone. Then a base strong enough to survive until morning.",
        "[determined] I punched my first tree... and the logs started dropping, one by one.",
        "[curious] Fun fact: you can punch wood with your bare hands. It just takes a while.",
        "[excited] Four logs became sixteen planks. Planks became a crafting table, then sticks... then my very FIRST pickaxe.",
        "[curious] A wooden pickaxe isn't much. But it can mine stone... and stone changes everything.",
    ]),
    (2, "THE CAVE", [
        "[suspenseful] Nine minutes left. I found a cave at the edge of the forest... dark, cold, and quiet.",
        "[whispering] Way too quiet.",
        "[determined] I mined straight in. Stone... more stone... then black specks in the wall. COAL!",
        "[excited] Coal means torches, and torches mean light. And light is the one thing monsters hate.",
        "[curious] Then, deeper in, I found something that didn't belong there. An old wooden chest.",
        "[suspenseful] I opened it slowly... and inside were six blocks of TNT.",
        "[excited] Someone had left a gift. Or a warning. Either way, I took all six.",
        "[whispering] And right next to the chest, glowing in the torchlight... a single DIAMOND.",
        "[excited] I didn't even know what to say. I just grabbed it and ran.",
    ]),
    (3, "THE BASE", [
        "[determined] Seven minutes left. Time to build.",
        "[excited] Cobblestone walls, a plank floor, and a roof that nothing could drop through.",
        "[curious] The trick is height. Make a wall two blocks tall... and zombies can't climb over it.",
        "[determined] Then a watchtower in the corner, so I could see them coming.",
        "[excited] From up there, I could see the whole forest... and the sun, getting lower and lower.",
        "[curious] Around the whole base, I dug a trench and filled it with water. A moat, to slow them down.",
        "[excited] Torches on every corner. One final block on the roof... and the base was DONE.",
    ]),
    (4, "NIGHTFALL", [
        "[suspenseful] Right on time, the sun touched the horizon... and the sky started turning orange.",
        "[suspenseful] Then purple. Then dark.",
        "[whispering] I climbed the tower... and waited.",
        "[suspenseful] The first zombie crawled out of the ground... then another... then five more.",
        "[panicked] They marched straight for my walls, groaning. But the moat held them back.",
        "[relieved] For a moment, I actually thought the plan had worked.",
        "[curious] They kept bumping into the water, turning around, and trying again. It was almost funny.",
        "[whispering] And then I heard it. That sound. Ssssss.",
        "[panicked] A creeper had crept right up to the back wall... and nothing I built could stop it.",
        "[shouting] BOOM! The wall EXPLODED, and suddenly there was a hole straight into my base!",
        "[panicked] Zombies started pouring in. No sword. Half my health. And the night was only getting started.",
    ]),
    (5, "THE TRAP", [
        "[determined] So I remembered the chest. Six blocks of TNT.",
        "[curious] I needed every monster in one place. And I had the one thing they all wanted... me.",
        "[determined] I jumped down from the tower and sprinted into the open field. The whole horde turned to follow.",
        "[panicked] Zombies on my left. Creepers on my right. One of them clipped me. Half a heart left.",
        "[determined] I dropped the TNT in a line... one, two, three, four, five, six... and ran for the wall.",
        "[suspenseful] The horde walked right onto the TNT. I lit the fuse... and dove behind the wall.",
        "[shouting] Six blocks of TNT went off at once, and the creepers in the middle exploded TOO!",
        "[shouting] It was the biggest explosion I had ever seen. Dirt, stone, monsters... everything went flying!",
        "[relieved] When the smoke cleared, the field was one giant crater. And the horde... was gone.",
        "[excited] Six blocks of TNT. Zero monsters left standing. Best trade I have EVER made.",
    ]),
    (6, "SUNRISE", [
        "[relieved] A few zombies were still stumbling around... but then the sun came up.",
        "[excited] And zombies burn in sunlight! One after another, they caught fire... until the field was quiet.",
        "[curious] Creepers don't burn, by the way. But the TNT had already taken care of them.",
        "[relieved] I survived. My first night. Half a heart... but I SURVIVED.",
        "[relieved] I stood on top of the tower and just watched the sunrise for a while. I felt like I had earned it.",
        "[determined] I patched the hole, added a second wall, and turned that crater into a farm.",
        "[excited] Day one, done. A base, a watchtower, a moat... and one very lucky diamond.",
        "[curious] And that diamond? I'm saving it for something special.",
        "[excited] What should I build with it? Tell me in the comments... and I'll see you on day two!",
    ]),
]

LINES = [l for _, _, ls in CHAPTERS for l in ls]
CHAPTER_START_LINE = []
_i = 0
for _, _, ls in CHAPTERS:
    CHAPTER_START_LINE.append(_i)
    _i += len(ls)

# extra silence (seconds) after a line, keyed by the line's text start -> resolved to global index
_HOLDS = {
    "[excited] I spawned in a brand new world": 0.8,
    "[excited] Four logs became sixteen planks": 2.2,
    "[excited] I didn't even know what to say": 1.0,
    "[determined] Seven minutes left": 2.5,
    "[excited] Torches on every corner": 3.0,
    "[suspenseful] Then purple. Then dark.": 2.0,
    "[whispering] And then I heard it": 1.5,
    "[shouting] BOOM! The wall": 1.5,
    "[suspenseful] The horde walked right onto": 1.0,
    "[shouting] It was the biggest explosion": 3.0,
    "[excited] And zombies burn in sunlight": 2.0,
    "[relieved] I stood on top of the tower": 2.0,
    "[excited] What should I build with it?": 6.0,
}
HOLDS = {}
for _k, _v in _HOLDS.items():
    _idx = [i for i, l in enumerate(LINES) if l.startswith(_k)]
    assert len(_idx) == 1, _k
    HOLDS[_idx[0]] = _v
