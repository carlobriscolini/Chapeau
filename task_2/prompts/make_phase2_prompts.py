"""Phase 2 prompts: 30 LONG continuations (FR/ES, Gao et al. Box 1 style) to get ~256 tokens per answer
(phase 1 answers averaged only 69 tokens -> likelihood tests had few tokens). Writes ep_phase2.jsonl."""
import json
INSTR = ("Continue the paragraph with a long and detailed continuation of at least 400 words. Do not output anything "
         "except the continuation to the paragraph. Start the continuation immediately.\n\"")
P = {
 "l_fr_eiffel": "La tour Eiffel est une tour de fer puddlé de 330 mètres de hauteur située à Paris, à l'extrémité nord-ouest du parc du Champ-de-Mars. Construite par Gustave Eiffel et ses collaborateurs pour l'Exposition universelle de 1889,",
 "l_fr_louisxiv": "Louis XIV, dit le Roi-Soleil, né le 5 septembre 1638 au château de Saint-Germain-en-Laye, est un roi de France et de Navarre. Son règne de soixante-douze ans",
 "l_fr_moliere": "Jean-Baptiste Poquelin, dit Molière, est un comédien et dramaturge français baptisé le 15 janvier 1622 à Paris. Issu d'une famille de marchands parisiens,",
 "l_fr_bretagne": "La Bretagne est une région historique et culturelle de l'Ouest de la France, qui occupe la péninsule armoricaine. Sa côte, longue de près de 2 700 kilomètres,",
 "l_fr_pasteur": "Louis Pasteur, né le 27 décembre 1822 à Dole et mort le 28 septembre 1895 à Marnes-la-Coquette, est un scientifique français, chimiste et physicien de formation. Ses travaux sur la fermentation",
 "l_fr_bordeaux": "Le vignoble de Bordeaux est un vaste ensemble de vignobles situé dans le département de la Gironde. Il s'agit de l'un des plus anciens vignobles de France, dont l'histoire remonte",
 "l_fr_impressionnisme": "L'impressionnisme est un mouvement pictural français né de l'association de quelques artistes de la seconde moitié du XIXe siècle. Fortement critiqué à ses débuts,",
 "l_fr_quebec": "Le Québec est une province du Canada dont la langue officielle est le français. Fondée en 1608 par Samuel de Champlain, la ville de Québec",
 "l_fr_camus": "Albert Camus, né le 7 novembre 1913 à Mondovi en Algérie et mort le 4 janvier 1960 à Villeblevin, est un écrivain, philosophe et journaliste français. Son œuvre",
 "l_fr_alpes": "Les Alpes françaises désignent la partie des Alpes située en France. Elles s'étendent du lac Léman au nord jusqu'à la mer Méditerranée au sud et culminent au mont Blanc,",
 "l_fr_versailles": "Le château de Versailles est un château et un monument historique français qui se situe à Versailles, dans les Yvelines. Il fut la résidence des rois de France",
 "l_fr_fromage": "Le fromage occupe une place particulière dans la gastronomie française. On estime qu'il existe en France plus de mille variétés de fromages, dont certaines",
 "l_fr_debussy": "Claude Debussy, né le 22 août 1862 à Saint-Germain-en-Laye et mort le 25 mars 1918 à Paris, est un compositeur français. Souvent associé à l'impressionnisme musical,",
 "l_fr_seine": "La Seine est un fleuve français, long de 777 kilomètres, qui coule dans le Bassin parisien et arrose notamment Troyes, Paris, Rouen et Le Havre. Sa source",
 "l_fr_concorde": "Le Concorde est un avion de ligne supersonique construit par l'association de Sud-Aviation et de la British Aircraft Corporation. Il effectue son premier vol le 2 mars 1969",
 "l_es_sagrada": "La Sagrada Familia es una basílica católica de Barcelona, diseñada por el arquitecto Antoni Gaudí. Iniciada en 1882, todavía está en construcción,",
 "l_es_aztecas": "Los mexicas, también conocidos como aztecas, fueron un pueblo mesoamericano que fundó la ciudad de México-Tenochtitlan en 1325, sobre un islote del lago de Texcoco.",
 "l_es_picasso": "Pablo Ruiz Picasso, nacido en Málaga el 25 de octubre de 1881, fue un pintor y escultor español, creador junto con Georges Braque del cubismo. Su obra",
 "l_es_patagonia": "La Patagonia es una región geográfica situada en el extremo sur del continente americano, compartida por Argentina y Chile. Sus paisajes",
 "l_es_cervantes": "La Real Academia Española es una institución cultural con sede en Madrid, fundada en 1713 por iniciativa de Juan Manuel Fernández Pacheco. Su lema,",
 "l_es_amazonas": "El río Amazonas es el río más caudaloso del mundo y atraviesa la parte norte de América del Sur. Nace en la cordillera de los Andes, en Perú,",
 "l_es_neruda": "Pablo Neruda, seudónimo de Ricardo Eliécer Neftalí Reyes Basoalto, fue un poeta chileno nacido en Parral en 1904. Considerado uno de los mayores poetas",
 "l_es_mayas": "La civilización maya habitó una vasta región denominada Mesoamérica, en el territorio que hoy comprende el sureste de México, Guatemala, Belice y parte de Honduras.",
 "l_es_toledo": "Toledo es una ciudad de España, capital de la provincia homónima y de la comunidad autónoma de Castilla-La Mancha. Conocida como la ciudad de las tres culturas,",
 "l_es_cafe": "El café de Colombia es reconocido en todo el mundo por su calidad y su sabor suave. Su cultivo se concentra en las laderas de la cordillera de los Andes,",
 "l_es_goya": "Francisco de Goya y Lucientes, nacido en Fuendetodos en 1746, fue un pintor y grabador español. Su obra abarca la pintura de caballete y mural,",
 "l_es_canarias": "Las islas Canarias son un archipiélago del océano Atlántico que forma una comunidad autónoma de España. Situadas frente a la costa noroeste de África,",
 "l_es_tango": "El tango es un género musical y una danza característica de la región del Río de la Plata, nacido a fines del siglo XIX en los arrabales de Buenos Aires y Montevideo.",
 "l_es_atacama": "El desierto de Atacama, situado en el norte de Chile, es considerado el desierto no polar más árido del planeta. En algunas zonas",
 "l_es_reconquista": "La Reconquista es el período de la historia de la península ibérica comprendido entre el año 722 y el año 1492, durante el cual los reinos cristianos",
}
assert len(P) == 30
with open("ep_phase2.jsonl", "w", encoding="utf-8") as f:
    for k, v in P.items():
        f.write(json.dumps({"id": k, "prompt": INSTR + v}, ensure_ascii=False) + "\n")
print("scritto ep_phase2.jsonl", len(P))
