package com.spring.biz.common;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.springframework.stereotype.Component;

@Component
@Aspect
public class Before2 {
	//포인트컷들만 모아놓을 수도 있음
//	@Before("PointCutCommon.allPointCut()")
//	public void beforeLog(JoinPoint jp) {
//		//클라이언트의 모든 요청정보를 담아서 스프링컨테이너가 제공하는 객체
//		String method = jp.getSignature().getName();
//		Object[] args = jp.getArgs();
//		
//		System.out.println("[사전처리] "+method+"()메서드 args정보: "+args[0].toString());
//		
		//System.out.println("[사전처리] - 비즈니스로직 수행시 동작");
//	}
}
