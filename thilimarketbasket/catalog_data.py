"""The hand-written raw material for the catalogue: Indian grocery products, brands, pack sizes, and how
much each one sells.

The weights are relative, not percentages: a product with weight 20 sells twice as much as one with weight 10
in the same category. Sales share flows down four levels, and each level is normalised on its own:

    category  ->  product within the category  ->  brand within the product  ->  pack size within the product

so a big category has a big share, a staple (Milk, Onion) outsells a rarity (Tea Bags), a leading brand
outsells a small one, and small packs outsell a 5 kg sack. These are our own estimates of a typical Indian
neighbourhood shop, not measured data. Edit them freely: change a number and the shop changes.

The brands are invented. Any match with a real brand is a coincidence. We use invented brands on purpose: a
language model that has read the internet already "knows" that real brands and real products go together,
and that prior knowledge would blur what it learns from the baskets.

Each product is a tuple: (name, weight) or (name, weight, {size: weight}) when the product has its own pack
sizes. Fresh produce is sold loose, so it has no brands. Sachets and small rupee packs are here on purpose:
they are a large part of Indian retail.
"""

CATEGORIES = {
    "Staples": {
        "weight": 14,
        "brands": {"Kaveri Valley": 8, "Surya Grains": 5, "Maitri Foods": 3, "Shubham Select": 2, "Annadata": 1.5, "Pragati": 1},
        "sizes": {"500 g": 3, "1 kg": 6, "5 kg": 2},
        "products": [
            ("Basmati Rice", 8), ("Sona Masoori Rice", 14), ("Wheat Flour (Atta)", 20), ("Maida", 5),
            ("Semolina (Rava)", 6), ("Besan", 6), ("Rice Flour", 4), ("Ragi Flour", 3),
            ("Poha", 7, {"500 g": 4, "1 kg": 3}), ("Oats", 4, {"500 g": 4, "1 kg": 3}),
            ("Vermicelli", 3, {"200 g": 4, "500 g": 3}), ("Sago (Sabudana)", 2, {"250 g": 4, "500 g": 3}),
            ("Salt", 20, {"1 kg": 6, "5 kg": 1}), ("Jaggery", 5, {"500 g": 4, "1 kg": 3}),
        ],
    },
    "Pulses": {
        "weight": 8,
        "brands": {"Surya Grains": 8, "Kaveri Valley": 5, "Annadata": 3, "Maitri Foods": 2, "Shubham Select": 1.5, "Pragati": 1},
        "sizes": {"500 g": 3, "1 kg": 6, "5 kg": 2},
        "products": [
            ("Toor Dal", 20), ("Moong Dal", 10), ("Chana Dal", 8), ("Urad Dal", 8), ("Masoor Dal", 7),
            ("Rajma", 4), ("Kabuli Chana", 4), ("Green Moong", 3), ("Black Chana", 3), ("Lobia", 2),
        ],
    },
    "Oils and Ghee": {
        "weight": 7,
        "brands": {"Golden Drop": 8, "Dhara Fresh": 5, "Suryamukhi": 3, "Ruchira": 2, "Amrit Kalash": 1.5, "Navkar": 1},
        "sizes": {"500 ml": 4, "1 L": 6, "5 L": 1.5},
        "products": [
            ("Sunflower Oil", 20), ("Groundnut Oil", 8), ("Mustard Oil", 6), ("Coconut Oil", 5), ("Rice Bran Oil", 4),
            ("Ghee", 8, {"200 g": 4, "500 g": 4, "1 kg": 2}), ("Vanaspati", 2, {"500 g": 4, "1 kg": 3, "5 kg": 1}),
        ],
    },
    "Spices": {
        "weight": 9,
        "brands": {"Masala Mantra": 8, "Rasoi Royale": 5, "Swad Sagar": 3, "Kesari": 2, "Ruchira": 1.5, "Tadka Tales": 1},
        "sizes": {"sachet 10 g": 5, "100 g": 4, "200 g": 2},
        "products": [
            ("Turmeric Powder", 14), ("Red Chilli Powder", 14), ("Coriander Powder", 12), ("Cumin Seeds", 10),
            ("Mustard Seeds", 10), ("Garam Masala", 8), ("Sambar Powder", 6), ("Rasam Powder", 4), ("Black Pepper", 4),
            ("Cardamom", 3), ("Chaat Masala", 3), ("Biryani Masala", 3),
            ("Asafoetida (Hing)", 5, {"sachet 5 g": 5, "50 g": 2}), ("Tamarind", 5, {"200 g": 4, "500 g": 3}),
        ],
    },
    "Dairy and Eggs": {
        "weight": 12,
        "brands": {"Gokul Fresh": 8, "Nandini Pure": 5, "Amulya": 3, "Kamadhenu": 2, "Dairy Dew": 1},
        "sizes": {"200 g": 4, "500 g": 3},
        "products": [
            ("Milk", 30, {"500 ml": 6, "1 L": 4}), ("Curd", 14, {"200 g": 5, "500 g": 3}),
            ("Paneer", 6, {"200 g": 5, "500 g": 2}), ("Butter", 6, {"100 g": 5, "500 g": 2}),
            ("Cheese Slices", 3, {"10 slices": 4, "20 slices": 2}), ("Eggs", 12, {"6 pack": 5, "12 pack": 4, "30 tray": 2}),
            ("Buttermilk", 4, {"200 ml": 5, "1 L": 2}),
        ],
    },
    "Bakery and Breakfast": {
        "weight": 6,
        "brands": {"Crunchy Crust": 8, "Baker's Basket": 5, "Sunrise Bites": 3, "Morning Crisp": 2, "Oven Fresh": 1},
        "sizes": {"100 g": 4, "200 g": 3, "400 g": 2},
        "products": [
            ("Bread", 20, {"400 g": 6, "800 g": 2}), ("Glucose Biscuits", 12), ("Cream Biscuits", 8),
            ("Rusk", 4, {"200 g": 4, "400 g": 2}), ("Cornflakes", 3, {"250 g": 4, "500 g": 3}),
            ("Jam", 3, {"200 g": 4, "500 g": 3}), ("Peanut Butter", 2, {"200 g": 4, "500 g": 3}),
        ],
    },
    "Snacks and Instant Food": {
        "weight": 9,
        "brands": {"Crunch Club": 8, "Chatpata": 5, "Snack Shala": 3, "Tasty Tales": 2, "Munch Mela": 1.5, "Bhel Bazaar": 1},
        "sizes": {"Rs 5 pack": 6, "Rs 10 pack": 5, "150 g": 2},
        "products": [
            ("Namkeen Mixture", 8), ("Potato Chips", 12), ("Instant Noodles", 14), ("Popcorn", 2), ("Murukku", 3),
            ("Banana Chips", 3), ("Roasted Peanuts", 4), ("Ready Idli Mix", 2),
            ("Dosa Batter", 6, {"500 g": 5, "1 kg": 3}), ("Pasta", 3, {"200 g": 4, "500 g": 3}),
        ],
    },
    "Beverages": {
        "weight": 8,
        "brands": {"Chai Chaska": 8, "Kaapi Kingdom": 5, "Sip Sagar": 3, "Amrit Pey": 2, "Thanda Taza": 1.5, "Brew Bazaar": 1},
        "sizes": {"250 g": 4, "500 g": 3},
        "products": [
            ("Tea Powder", 20, {"250 g": 5, "500 g": 4, "1 kg": 1.5}),
            ("Coffee Powder", 8, {"50 g": 4, "100 g": 4, "200 g": 2}),
            ("Instant Coffee Sachet", 6, {"sachet": 6, "strip of 10": 3}),
            ("Sugar", 20, {"1 kg": 6, "5 kg": 2}), ("Tea Bags", 2, {"25 bags": 4, "100 bags": 2}),
            ("Health Drink Powder", 4, {"200 g": 4, "500 g": 3}), ("Fruit Juice", 4, {"200 ml": 5, "1 L": 3}),
            ("Soft Drink", 8, {"250 ml": 5, "600 ml": 4, "2 L": 2}), ("Coconut Water", 2, {"200 ml": 5, "1 L": 2}),
            ("Soda", 3, {"600 ml": 4, "2 L": 2}),
        ],
    },
    "Fresh Produce": {
        "weight": 16,
        "brands": {},  # sold loose, no brand
        "sizes": {"500 g": 4, "1 kg": 5},
        "products": [
            ("Onion", 30), ("Potato", 28), ("Tomato", 28), ("Green Chilli", 14), ("Ginger", 10), ("Garlic", 10),
            ("Carrot", 6), ("Cabbage", 4), ("Cauliflower", 4), ("Brinjal", 5), ("Okra (Bhindi)", 4), ("Spinach", 3),
            ("Apple", 3), ("Coriander Leaves", 14, {"bunch": 1}), ("Curry Leaves", 8, {"bunch": 1}),
            ("Banana", 8, {"dozen": 1}), ("Lemon", 8, {"250 g": 5, "500 g": 3}),
        ],
    },
    "Household and Cleaning": {
        "weight": 5,
        "brands": {"Sparkle Home": 8, "Shuddh": 5, "Ghar Ghar": 3, "Nirmal": 2, "Fresh Fort": 1},
        "sizes": {"500 ml": 4, "1 L": 3},
        "products": [
            ("Detergent Powder", 12, {"500 g": 5, "1 kg": 4, "4 kg": 1}), ("Detergent Bar", 8, {"1 bar": 5, "pack of 4": 3}),
            ("Dishwash Liquid", 6), ("Dishwash Bar", 8, {"1 bar": 5, "pack of 3": 3}),
            ("Floor Cleaner", 3, {"500 ml": 4, "2 L": 2}), ("Toilet Cleaner", 3, {"500 ml": 4, "1 L": 3}),
            ("Phenyl", 2, {"500 ml": 4, "2 L": 2}), ("Agarbatti (Incense)", 8, {"pack of 20": 5, "pack of 100": 2}),
            ("Matchbox", 4, {"pack of 10": 1}), ("Candles", 2, {"pack of 6": 1}), ("Garbage Bags", 2, {"pack of 30": 1}),
            ("Tissue Roll", 2, {"1 roll": 4, "pack of 4": 2}),
        ],
    },
    "Personal Care": {
        "weight": 5,
        "brands": {"Glow Garden": 8, "Rupa Rani": 5, "Pure Petal": 3, "Sundar Sutra": 2, "Tulsi Tara": 1},
        "sizes": {"50 g": 3, "100 g": 4, "200 g": 2},
        "products": [
            ("Shampoo Sachet", 14, {"Rs 1 sachet": 6, "Rs 2 sachet": 5, "strip of 12": 2}),
            ("Shampoo Bottle", 4, {"180 ml": 4, "340 ml": 2}), ("Soap Bar", 14, {"1 bar": 5, "pack of 4": 3}),
            ("Toothpaste", 12), ("Toothbrush", 3, {"1 brush": 5, "pack of 3": 2}),
            ("Hair Oil", 6, {"100 ml": 5, "200 ml": 3, "500 ml": 1}), ("Face Cream", 3), ("Talcum Powder", 3),
            ("Sanitary Pads", 4, {"pack of 8": 4, "pack of 20": 2}), ("Razor", 2, {"1 razor": 5, "pack of 5": 2}),
            ("Body Lotion", 2, {"100 ml": 4, "400 ml": 2}), ("Deodorant", 2, {"100 ml": 4, "150 ml": 3}),
        ],
    },
    "Baby and Stationery": {
        "weight": 2,
        "brands": {"Little Lotus": 8, "Tiny Tots": 5, "Nanhe Kadam": 3, "Kids Korner": 1},
        "sizes": {"pack of 1": 2, "pack of 5": 1},
        "products": [
            ("Diapers", 4, {"pack of 20": 4, "pack of 56": 2}), ("Baby Cereal", 2, {"200 g": 4, "400 g": 2}),
            ("Baby Wipes", 2, {"pack of 30": 4, "pack of 72": 2}), ("Baby Soap", 2, {"1 bar": 5, "pack of 3": 3}),
            ("Notebook", 6), ("Ball Pen", 5, {"pack of 5": 4, "pack of 10": 2}),
            ("Batteries (AA)", 3, {"pack of 4": 4, "pack of 10": 2}),
        ],
    },
}
