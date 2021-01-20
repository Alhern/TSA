from analyzer import predict_this

# Quelques exemples...

print("\tRANDOM STRINGS:")
predict_this("This software has too many bugs, are you mocking me?")
predict_this("I'm surprised! It's a pretty interesting program")
predict_this("it's hard to believe this happened, what a shame")
predict_this("i'm relaxing with my cat on my lap, having the sweetest weekend")
predict_this("the cat is being very annoying...")
predict_this("even though she's clingy, she's a good girl")
predict_this("I don't feel better after taking this medication")
predict_this("I do feel better after taking this medication, it works!")
predict_this("I can't stand the cold weather!!")
predict_this("Cold weather isn't so bad, I can stay home and watch Spongebob with a nice cup of cocoa")


# Résultats attendus :

# NEGATIVE: {This software has too many bugs, are you mocking me?}
# POSITIVE: {I'm surprised! It's a pretty interesting program}
# NEGATIVE: {it's hard to believe this happened, what a shame}
# POSITIVE: {i'm relaxing with my cat on my lap, having the sweetest weekend}
# NEGATIVE: {the cat is being very annoying...}
# POSITIVE: {even though she's clingy, she's a good girl}
# NEGATIVE: {I don't feel better after taking this medication}
# POSITIVE: {I do feel better after taking this medication, it works!}
# NEGATIVE: {I can't stand the cold weather!!}
# POSITIVE: {Cold weather isn't so bad, I can stay home and watch Spongebob with a nice cup of cocoa}


print("\n\n\tCORPUS RELATED STRINGS")
print("\tSUPPOSED TO BE POSITIVE:")
predict_this("Cyberpunk 2077 is honestly the best game I've played in my life, all the b*tchn and moaning about glitches is the biggest joke I've heard, almost 40 hours in and I've only experienced like 7 minor background glitches, that's less than most games but we don't talk about that, huh?")
predict_this("Don't care what anyone says, 6 hours in on Xbox One S (yes, the ol stormtrooper from 2016) #Cyberpunk2077 is freaking amazing!!! 😍 The pacing, the immersion... twists and turns. I don't think I've ever been taken to another world in any game quite like this. Bravo @CDPROJEKTRED")
predict_this("I accepted the fact that it won't run as beautifully as it should in my old rig so I made my peace with that and moved on. I am now focused on the gigs and quest and I must say just like the rest of my RPGs this one got me hooked as well. Maybe a little more fixes and it's nova.")
predict_this("Seriously the prettiest game I've ever played! #Shutterpunk2077 #Cyberpunk2077")
predict_this("Alright, I think those CYBERPUNK 2077 Hotfix patches are working, cause the game's a lot smoother now. Still has the environment rendering issues like crazy, but it's definitely better than release day. Think I might just be able to do a review on it very soon! ")
predict_this("I’ve played 15 hours of #Cyberpunk2077 on #XboxSeriesX and it’s been an absolute thrill ride.  I honestly still have a lot to learn in game and I feel like I’ve just scratched the surface. Feels slightly daunting, but addictive. Story  and graphics are really great.")
predict_this("The gigs and side missions are all top notch aswell, and even the random encounters are filled with bunch of cool / sad lore that you can read and discover so I've had a lot of fun exploring and doing as much as I can within #Cyberpunk2077 for my first run! :3")
predict_this("Had a lovely stream, #Cyberpunk2077 FINALLY got past the beginning section cause I’m crazy busy and don’t put enough time towards the game. Thank you all for watching with me. Had a chance to raid @HearneMariah shoot her a follow on twitch, she is awesome!")
predict_this("Finally finished #Cyberpunk2077! Loved everything about the story and gave me the same feeling I had with Fallout: New Vegas. Now I’m just going to wait for all the patches, so I can play it again and have a better experience.")
predict_this("@CyberpunkGame Love this game amazing who care about bugs thay get fixed one way or another makes the game interesting keep up the good work #CDProjektRed #Cyberpunk2077")
predict_this("I am having a lot of fun with driving in cockpit/first-person mode with both cars &amp; bikes. The detailing is nice &amp; the vehicle designs r amazing. #Cyberpunk2077")
predict_this("Couldn’t say better myself! The game is definitely worth it. I’ve only played a few hours in, cause I’m more in a mood for Skyrim this time of the year, but from what I saw, the only bags I had were funny ones I only laughed at. Played on #SeriesX.")
predict_this("I've gotta say that despite its obvious problems, #Cyberpunk2077 has done a really good job with its side characters, most of them have good and intriguing quests. I hope we get to see even more of #JudyAlvarez in maybe a future DLC? ;) @CDPROJEKTRED #PS4share #videogames")
predict_this("Being impeached not once, but twice is HILARIOUS 🤣")

print("\n--------------------------------------------------\n")

print("\tSUPPOSED TO BE NEGATIVE:")
predict_this("@AskPlayStation I’ve requested a cyberpunk refund 2 weeks ago. I’ve deleted from my system and couldn’t re-download it if I wanted to since you pulled it from the store. Why are you withholding refunds? #Cyberpunk2077 #PSN")
predict_this("Remember folks, #Cyberpunk2077 looks THIS BAD on consoles. I'alright a last oneve no idea why I even play it. All aboard the hate bandwagon 🤣🤣😀 #choochoo https://t.co/Bp7KojTB8t")
predict_this("The game fucking crashes, during the credits.")
predict_this("#Cyberpunk2077 What the actual fuck happened to this game? When it was announced, we were hyped. But the bugs. THE GODDAMNED BUGS! Jesus!! Didn't anybody in the development cycle checked this game before releasing it?! What *were* they thinking?!")
predict_this("Awww, sucks... I tried to sleep at Judy's apartment but couldn't... Not a bug but definitely bigger bummer than some bugs I've encountered in #Cyberpunk2077")
predict_this("boring story, dumb AI, meaningless side missions")
predict_this("The glitch in #Cyberpunk2077 fckin sucks yo. I’m doing my brawls, freeze up then I’m dead 🤬")
predict_this("Hit the first part of Cyber Punk where I thought, ah shit. Its broken. During the heist I got stuck in a table 😂😂 #Cyberpunk2077")
predict_this("There’s no denying that #Cyberpunk2077  is a technical disappointment, especially on last gen consoles and it certainly shouldn’t have been released  in it’s unfinished condition but personally Alien colonial marines  was worse and surely more anti consumer")
predict_this("Find out how \"crunching\" long hours and working 6 days a week on a software project wastes developers' skills and will kill even the most anticipated of products. Bad Software Engineering KILLED Cyberpunk 2077’s Release\" by @davefarley77 #Cyberpunk2077 #Cyberpunk2077bugs")
predict_this("@CyberpunkGame #cyberpunk2077  was suppose to be the game of the century instead #CDProjektRed executives lost that vision and decided to deceive consumers and release a broken unplayable game, that only works properly on PC if you have a 3080. @CDPROJEKTRED 8 years you tried and then failed")
predict_this("@Swing_Knowles @JakeBaldino LoL damn, yeah, I’m about to beat #Cyberpunk2077 on PC since I have over 40+ hours in it and this was such a boring, meaningless game. I regret buying into all the hype when this game has nothing good about it except for the graphics")
predict_this("This game is so fucking stupid #Cyberpunk2077")
predict_this("I spent just shy of 60 Hours on #Cyberpunk2077 - despite all the glitches, bugs and a total of 17 crashes, I loved so much of it - apart from the ending.  It felt so underwhelming and rushed and the relationships I built seemed mostly meaningless.. now I just feel sad :(")
predict_this("It is a sad and unfortunate historical moment, but an absolutely necessary one.")
predict_this("being impeached twice is so fucking embarrassing bye")

print("\n--------------------------------------------------\n")

# On constate quelques classifications différentes entre les 2 modèles dans pretrained pour les tweets suivants qui ne semblent pas assez clairs pour notre analyseur, le modèle 1 semble  légèrement plus doué pour identifier les sentiments négatifs que le modèle 2 mais dans les 2 cas, il semble qu'il y ait un biais positif dans les 2 modèles

print("\nMISCLASSIFIED & CONFUSING EXAMPLES")
print("\nShould be neg:"), predict_this("@_ShauryaChawla It’s improved a bit, yeah, but I feel this game was so overhyped. It’s not a good game at all in my book, boring story, dumb AI, meaningless side missions. Just not fun at all after the first few hours #Cyberpunk2077")

# le mot 'repetitive' n'est pas assez négatif pour les modèles
print("\nShould be neg:"), predict_this("this game can get pretty repetitive at times #PS5Share, #Cyberpunk2077")

# 'here is the backstabbing traitors' est négatif pour le modèle 2, mais "'s" le rend positif pour ce modèle... Why?
print("\nShould be neg:"), predict_this("Here's the BACKSTABBING TRAITORS")

# trop de fautes, trop de contractions, parfait pour rater l'analyse, négatif pour le modèle 1, positif pour le modèle 2
print("\nShould be neg:"), predict_this("Cause everyone hates it... imma try #Cyberpunk2077 not saying thats gonna make me like it im most likly gonna find it just as boring but imma give it the good ol collage try...")

# pas assez clair pour le modèle 2 (pos), négatif pour le modèle 1.
# Ce tweet est subtil, est ce qu'être le nouveau Peter Molyneux est positif ?
# Est ce que Fable 3 est positif ? La machine ne comprend rien à ce charabia,
# le coeur du sentiment devrait se trouver dans "under delivers", mais "so much promise" peut nous rendre confus.
print("\nShould be neg:"), predict_this("CDPR is the new Peter Molyneux of gaming. So much promise of all these wild systems and then heavily under delivers. CP2077 kinda feels like Fable 3 all over again. #Cyberpunk2077 #gaming")

# Les modèles ratent la subtilité du language
print("\nShould be neg:"), predict_this("The worst thing about an Impeachment trial. Now we gotta listen to Jonathan Turley and Alan Dershowitz making believe they're relevant ...")

# Modèle 2 se trompe sur "cant get enough"
print("\nShould be pos:"), predict_this("Been playing #Cyberpunk2077 since release. Just cant get enough of this world! #urbanexploration  #freerunning #urbanexploration #urbex  #Cyberpunk2077PhotoMode")

# Modèle 1 se trompe
print("\nShould be neg:"), predict_this("It's a jumbled mess of things we've seen 1000 times before. Put away the golden straw.")

