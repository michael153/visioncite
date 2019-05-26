import datetime

class Citation:
	def __init__(self):
		self.title = ""
		self.authors = []
		self.date_month = ""
		self.date_day = ""
		self.date_year = ""
		self.publisher = ""
		self.url = ""

	def __init__(self, title="", authors=[], date=("","",""), publisher="", url=""):
		self.title = title
		self.authors = authors
		self.date_month = date[0]
		self.date_day = date[1]
		self.date_year = date[2]
		self.publisher = publisher
		self.url = url

	def get_name_splice(self, author):
		if " " not in author:
			return ('', '', author)
		else:
			first = author.find(" ")
			last = author.rfind(" ")
			if author.count(" ") < 2:
				return (author[:first],
					'', author[last + 1:])
			else:
				return (author[:first],
					author[first + 1: last],
					author[last + 1:])

	# def get_middle_initial(self, author):
	# 	if author.count(" ") < 2:
	# 		return None
	# 	else:
	# 		first = author.find(" ")
	# 		last = author.rfind(" ")
	# 		return author[first + 1: last][0]

	# def get_last_name(self, author):
	# 	if " " not in author:
	# 		return author
	# 	else:
	# 		return author[author.rfind(" ") + 1:]

	def format_mla(self, accessed_today=False):
		enquote = lambda t: "\"" + t + "\""
		italicize = lambda t: "<i>" + t + "</i>"
		ret = ""
		if self.authors:
			self.authors.sort(key=lambda a: self.get_name_splice(a)[2])
			n = len(self.authors)
			if n > 3:
				ret += self.get_name_splice(self.authors[0])[2] + " et al. "
			elif n >= 1:
				ret += ", ".join([self.get_name_splice(a)[2] for a in self.authors[:-1 if n == 3 else None]])
				if n == 3:
					ret += ", and " + self.get_name_splice(self.authors[-1])[2] + ". "
				else:
					ret += ". "
			else:
				sole = self.authors[0]
				if " " in sole:
					last_name = get_name_splice(sole)[2]
					ret += last_name + ", " + sole[:sole.find(last_name) - 1] + ". "
				else:
					ret += sole + ". "
		if self.title:
			ret += enquote(self.title + ".") + " "
		is_date = self.date_year or self.date_month or self.date_day
		if self.publisher:
			ret += italicize(self.publisher)
			if is_date or accessed_today:
				ret += ", "
			else:
				ret += "."
		if is_date:
			date = [self.date_day, self.date_month, self.date_year]
			ret += " ".join([str(d) for d in date if d]) + ". "
		if accessed_today:
			ret += "Accessed " + datetime.datetime.now().strftime("%d %B %Y") + "."
		ret = ret.strip()
		return ret

	# def format_apa(self):
	# 	italicize = lambda t: "<i>" + t + "</i>"
	# 	ret = ""
	# 	if self.authors:
	# 		self.authors.sort(key=lambda a: self.get_name_splice(a)[2])
	# 		n = len(self.authors)
	# 		# ret += ", ".join([c + ", " + b[0] + ". " + a[0] + "." for (a, b, c) in get_name_splice(a)])
	# 		if n > 3:
	# 			ret += self.get_name_splice(self.authors[0]) + " et al. "
	# 		elif n >= 1:
	# 			ret += ", ".join([self.get_name_splice(a) for a in self.authors[:-1 if n == 3 else None]])
	# 			if n == 3:
	# 				ret += ", and " + self.get_name_splice(self.authors[-1]) + ". "
	# 			else:
	# 				ret += ". "
	# 		else:
	# 			sole = self.authors[0]
	# 			if " " in sole:
	# 				last_name = get_name_splice(sole)
	# 				ret += last_name + ", " + sole[:sole.find(last_name) - 1] + ". "
	# 			else:
	# 				ret += sole + ". "



c = Citation(title="On Global Warming and Financial Imbalances",
		authors=["John Milken", "Michael Wan", "John Doe", "John Simerlink", "Balaji Veeramani", "Andrew Kirrilov"],
		date=("May", "", 2007),
		publisher="New Perspectives Quarterly")

print(c.format_mla())



