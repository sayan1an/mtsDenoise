#if !defined(__APPENDER_H)
#define __APPENDER_H

MTS_NAMESPACE_BEGIN

/** \brief This class defines an abstract destination
 * for logging-relevant information
 */
class MTS_EXPORT_CORE Appender : public Object {
public:
	/// Append a line of text
	virtual void append(ELogLevel level, const std::string &pText) = 0;

	/**
	 * Process a progress message
	 * @param progress Percentage value in [0,100]
	 * @param name Title of the progress message
	 * @param formatted Formatted string representation of the message
	 * @param eta Estimated time until 100% is reached.
	 * @param ptr Custom pointer payload
	 */
	virtual void logProgress(Float progress, const std::string &name,
		const std::string &formatted, const std::string &eta, const void *ptr) = 0;

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~Appender() { }
};

/** \brief Appender implementation which writes to an
 * arbitrary C++ stream
 */
class MTS_EXPORT_CORE StreamAppender : public Appender {
public:
	/// Create a new stream appender
	StreamAppender(std::ostream *pStream);

	/// Create a new stream appender logging to a file
	StreamAppender(const std::string &pFilename);

	/// Append a line of text
	void append(ELogLevel level, const std::string &pText);

	/// Process a progress message
	void logProgress(Float progress, const std::string &name,
		const std::string &formatted, const std::string &eta,
		const void *ptr);

	/// Does this appender log to a file
	inline bool logsToFile() const { return m_isFile; }

	/// Return a string representation
	std::string toString() const;

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~StreamAppender();
private:
	std::ostream *m_stream;
	std::string m_fileName;
	bool m_isFile;
	bool m_lastMessageWasProgress;
};

/** \brief Appender implementation which writes to an
 * unbuffered file descriptor.
 */
class MTS_EXPORT_CORE UnbufferedAppender : public Appender {
public:
	/// Create a new appender
	UnbufferedAppender(int fd);

	/// Append a line of text
	void append(ELogLevel level, const std::string &pText);

	/// Process a progress message
	void logProgress(Float progress, const std::string &name,
		const std::string &formatted, const std::string &eta,
		const void *ptr);

	/// Return a string representation
	std::string toString() const;

	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~UnbufferedAppender();
private:
	int m_fd;
	bool m_lastMessageWasProgress;
};

MTS_NAMESPACE_END

#endif /* __APPENDER_H */