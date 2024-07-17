package com.spring.biz.board.impl;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import com.spring.biz.board.BoardVO;

@Repository
public class BoardDAOSpring {

	@Autowired
	private JdbcTemplate jdbcTemplate;
	
//	private final String BOARD_INSERT = 
//			"insert into board(title, writer, content) values (?, ?, ?)";
	private final String BOARD_INSERT = 
			"insert into board(seq, title, writer, content) values (?, ?, ?, ?)";
	private final String BOARD_UPDATE = 
			"update board set title=?, content=? where seq=?";
	private final String BOARD_DELETE = 
			"delete from board where seq=?";
	private final String BOARD_GET =
			"select * from board where seq=?";
	private final String BOARD_LIST =
			"select * from board order by seq desc";
	
	public void insertBoard(BoardVO vo) {
		jdbcTemplate.update(BOARD_INSERT, vo.getSeq(), vo.getTitle(), vo.getWriter(), vo.getContent());
	}
	public void updateBoard(BoardVO vo) {
		jdbcTemplate.update(BOARD_UPDATE, vo.getTitle(), vo.getContent(), vo.getSeq());
	}
	public void deleteBoard(BoardVO vo) {
		jdbcTemplate.update(BOARD_DELETE, vo.getSeq());
	}
	
	public BoardVO getBoard(BoardVO vo) {
		Object[] args = {vo.getSeq()};
		return jdbcTemplate.queryForObject(BOARD_GET, args, new BoardMapper());
	}
	
	public List<BoardVO> getBoardList(){
		return jdbcTemplate.query(BOARD_LIST, new BoardMapper());
	}
}
