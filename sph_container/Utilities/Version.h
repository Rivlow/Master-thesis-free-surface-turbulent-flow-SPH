#ifndef __Version_h__
#define __Version_h__

#define STRINGIZE_HELPER(x) #x
#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define WARNING(desc) message(__FILE__ "(" STRINGIZE(__LINE__) ") : Warning: " #desc)

#define GIT_SHA1 "e773d3e26655b32ab67a9ac7c48f6f2bf22f9429"
#define GIT_REFSPEC "refs/heads/master"
#define GIT_LOCAL_STATUS "CLEAN"

#define SPLISHSPLASH_VERSION "2.13.1"

#ifdef DL_OUTPUT

#endif

#endif
