package com.spring.biz.board;

import java.util.List;

import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.GenericXmlApplicationContext;

import com.spring.biz.board.impl.BoardService;


public class BoardServiceClient {
	public static void main(String[] args) {
		
		AbstractApplicationContext factory = new GenericXmlApplicationContext("applicationContext.xml");
		BoardService service = (BoardService)factory.getBean("boardService");
		
		//게시글 추가
		BoardVO vo = (BoardVO)factory.getBean("boardVO");
		vo.setTitle("임시제목");
		vo.setWriter("작자미상");
		vo.setContent("수요일입니다");
		
		service.insertBoard(vo);
		
		
		//게시글리스트 조회
		List<BoardVO> boardList =  service.getBoardList();
		for(BoardVO board : boardList) {
			System.out.println(">>"+board.toString());
		}
		
		//특정번호 조회
		
		BoardVO searchVO = (BoardVO)factory.getBean("boardVO");
		searchVO.setSeq(5);
		BoardVO resultVO = service.getBoard(searchVO);
		if(resultVO != null) {
			System.out.println("조회결과: "+resultVO.toString());
		}else {
			System.out.println("조회결과가 없습니다.");
		}
		
		//수정
		BoardVO updateVO = (BoardVO)factory.getBean("boardVO");
		updateVO.setSeq(5);
		updateVO.setTitle("수정할 제목");
		updateVO.setContent("수정할 내용");
		service.updateBoard(updateVO);
		
		//삭제
		BoardVO deleteVO = (BoardVO)factory.getBean("boardVO");
		deleteVO.setSeq(1);
		service.deleteBoard(deleteVO);
	}
}
