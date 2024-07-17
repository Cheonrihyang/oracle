package com.spring.biz.common;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;

//빈 등록
@Component("beforeJoin")
@Aspect
public class BeforeAdvice {
	

	//어떤 pointcut을 적용?
	@Before("allPointCut()")
	public void beforeLog(JoinPoint jp) {
		//클라이언트의 모든 요청정보를 담아서 스프링컨테이너가 제공하는 객체
		String method = jp.getSignature().getName();
		Object[] args = jp.getArgs();
		
		System.out.println("[사전처리] "+method+"()메서드 args정보: "+args[0].toString());
		
		//System.out.println("[사전처리] - 비즈니스로직 수행시 동작");
	}
}
