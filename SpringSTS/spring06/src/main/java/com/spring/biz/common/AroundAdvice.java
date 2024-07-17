package com.spring.biz.common;

import org.aspectj.lang.ProceedingJoinPoint;
import org.springframework.util.StopWatch;

public class AroundAdvice {
	public Object aroundLog(ProceedingJoinPoint pjp) throws Throwable{
		
		String method = pjp.getSignature().getName();
		StopWatch watch = new StopWatch();
		
		watch.start();
		Object obj = pjp.proceed();
		watch.stop();
		
		System.out.println(method+"() 메서드 수행시간: "+watch.getTotalTimeMillis()+"(ms)초");
		
		return obj;
		
	}
}
